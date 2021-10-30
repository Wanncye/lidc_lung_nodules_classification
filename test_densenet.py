import model.data_loader as data_loader
from model.threeDDensenet import DenseNet201
import numpy as np
import os


import torch

# image process
import cv2
from PIL import Image

img_save_dir = "imgs"
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir, exist_ok=True)

@torch.no_grad()
def test_densenet(ckpt_path: str):
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 2, data_dir="data/nodules3d_128_npy_no_same_patient_in_two_dataset", train_shuffle=False)
    # dataloaders = data_loader.fetch_N_folders_dataloader(test_folder=N_folder, types = ["train", "test"], batch_size = params.batch_size, data_dir=params.data_dir)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    model = DenseNet201().cuda()
    
    model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    model.eval()

    for dataloader_index, (data_batch, labels_batch, filename) in enumerate(test_dl):
        output_batch, _ = model(data_batch.cuda())
        output_batch = torch.softmax(output_batch, dim=1)
        save_img(data_batch, output_batch, labels_batch, dataloader_index)

    print("finished...")


def save_img(data_batch: torch.Tensor, output_batch: torch.Tensor, gt_label: torch.Tensor, batch_idx: int):
    """
    data_batch: (N, C, D, H, W)
    output_batch: (N, 2) 
    gt_label: (N, )
    """
    
    assert data_batch.ndim == 5 and gt_label.ndim == 1 and data_batch.shape[1] == 1
    data_batch = data_batch.squeeze(dim=1)  # (N, D, H, W)
    data_batch, gt = data_batch.cpu().numpy(), gt_label.cpu().numpy()
    pred_sc, pred_cls = output_batch.max(dim=1)
    pred_sc, pred_cls = pred_sc.cpu().numpy(), pred_cls.cpu().numpy()


    for i, (data, sc, label, gt) in enumerate(zip(data_batch, pred_sc, pred_cls, gt_label)):
        # data: (D, H, W)
        # label: scalar
        if label == 1:
            pred_color = (0, 0, 255)  # red
        else:
            pred_color = (0, 255, 0)  # green
        
        if gt == 1:
            gt_color = (0, 0, 255)  # red
        else:
            gt_color = (0, 255, 0)  # green

        img = data[(data.shape[0]-1) // 2]
        img = np.asarray(Image.fromarray(img).convert("RGB"))

        img = cv2.resize(img, dsize=(512, 512))
        cv2.putText(img, "Groud Truth: " + ("Malignant" if gt == 1 else "Benign"), (2, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, color=gt_color, fontScale=0.6)
        cv2.putText(img, "Predication: " + (f"Malignant - {sc:.4f}" if label == 1 else f"Benign - {sc:.4f}"), (2, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, color=pred_color, fontScale=0.6)

        cv2.imwrite(os.path.join(img_save_dir, f"{batch_idx}_{i}.jpeg"), img)
        
test_densenet('./experiments/densenet201_nomask/folder.0.FocalLoss_alpha_0.25.best.pth.tar')
