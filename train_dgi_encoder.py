import torch
from model.dgi_encoder import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats
import argparse

import model.data_loader as data_loader
from torch.optim.lr_scheduler import StepLR,MultiStepLR

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # M:16, 128, 4, 124, 124
        y_exp = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #16,64,1,1,1
        y_exp = y_exp.expand(-1, -1, 4, 124, 124)
        # M 是local_feature, y_exp是expend后的全局特征，M_prime是负样本
        y_M = torch.cat((M, y_exp), dim=1) #16, 192, 4, 124, 124
        y_M_prime = torch.cat((M_prime, y_exp), dim=1) #16, 192, 4, 124, 124

        Ej = -F.softplus(-self.local_d(y_M)).mean()   #真实样本
        Em = F.softplus(self.local_d(y_M_prime)).mean()#负样本对
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--max_epoch', default=100, type=int, help='max_epoch')
    parser.add_argument('--data_dir', default='./data/nodules3d_128_npy', type=str, help='data_dir')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning_rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    # torch.cuda.empty_cache()
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    data_dir = args.data_dir
    learning_rate = args.learning_rate

    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = batch_size, data_dir=data_dir)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    encoder = Encoder().to(device)
    loss_fn = DeepInfoMaxLoss().to(device)
    optim = Adam(encoder.parameters(), lr=learning_rate)
    loss_optim = Adam(loss_fn.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optim, milestones=[12,30,70], gamma=0.5)

    for epoch in range(max_epoch):
        train_loss = []
        with tqdm(total=len(train_dl)) as t:
            for i, (x, target, _) in enumerate(train_dl):
                x = x.to(device)
                optim.zero_grad()
                loss_optim.zero_grad()
                y, M = encoder(x) #M:16, 128, 4, 124, 124
                # rotate images to create pairs for comparison
                M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)   #即让最后一个样本调换到第一个样本的位置处，以此类推
                loss = loss_fn(y, M, M_prime)
                train_loss.append(loss.item())
                loss.backward()
                optim.step()
                t.set_description(str(epoch+1) + ' Loss: ' + str(stats.mean(train_loss[-20:])) + ' lr: ' + str(optim.param_groups[0]['lr']))
                t.update()

        scheduler.step()

        if epoch % 20 == 0:
            learning_rate = learning_rate * 0.5
            root = Path('./experiments/dgi')
            enc_file = root / Path('encoder_feature.256_epoch.' + str(epoch+1) + '.wgt')
            loss_file = root / Path('loss' + str(epoch+1) + '.wgt')
            enc_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), str(enc_file))
            # torch.save(loss_fn.state_dict(), str(loss_file))
