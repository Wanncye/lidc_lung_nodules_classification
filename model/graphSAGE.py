# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn.parameter import Parameter

# class MeanAggregator(nn.Module):
#     def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
#                  use_bias=False,
#                  seed=1024, **kwargs):
#         super(MeanAggregator, self).__init__()
#         self.units = units
#         self.neigh_max = neigh_max
#         self.concat = concat
#         self.dropout_rate = dropout_rate
#         self.l2_reg = l2_reg
#         self.use_bias = use_bias
#         self.activation = activation
#         self.seed = seed
#         self.input_dim = input_dim

#         self.neigh_weights = Parameter(torch.FloatTensor(self.input_dim, self.units))
#         nn.init.xavier_uniform_(self.neigh_weights.data, gain=1.414)
#         if self.use_bias:
#             self.bias = Parameter(torch.FloatTensor(self.units))
#             nn.init.xavier_uniform_(self.bias.data, gain=1.414)

#         self.dropout = nn.Dropout(self.dropout_rate)

#     def forward(self, inputs):
#         features, node, neighbours = inputs

#         node_feat = nn.embedding_lookup(features, node)
#         neigh_feat = nn.embedding_lookup(features, neighbours)

#         node_feat = self.dropout(node_feat)
#         neigh_feat = self.dropout(neigh_feat)

#         concat_feat = torch.concat([neigh_feat, node_feat], axis=1)
#         concat_mean = torch.reduce_mean(concat_feat, axis=1, keep_dims=False)

#         output = torch.matmul(concat_mean, self.neigh_weights)
#         if self.use_bias:
#             output += self.bias
#         if self.activation:
#             output = self.activation(output)

#         return output

#     def get_config(self):
#         config = {'units': self.units,
#                   'concat': self.concat,
#                   'seed': self.seed
#                   }

#         base_config = super(MeanAggregator, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# class PoolingAggregator(nn.Module):

#     def __init__(self, units, input_dim, neigh_max, aggregator='meanpooling', concat=True,
#                  dropout_rate=0.0,
#                  activation=nn.relu, l2_reg=0, use_bias=False,
#                  seed=1024, ):
#         super(PoolingAggregator, self).__init__()
#         self.output_dim = units
#         self.input_dim = input_dim
#         self.concat = concat
#         self.pooling = aggregator
#         self.dropout_rate = dropout_rate
#         self.l2_reg = l2_reg
#         self.use_bias = use_bias
#         self.activation = activation
#         self.neigh_max = neigh_max
#         self.seed = seed


#         self.dense_layers = nn.Linear(self.input_dim, self.input_dim)
#         self.relu = nn.ReLU()
#         self.neigh_weights = Parameter(torch.FloatTensor(self.input_dim* 2, self.output_dim))
#         nn.init.xavier_uniform_(self.neigh_weights.data, gain=1.414)

#         if self.use_bias:
#             self.bias = Parameter(torch.FloatTensor(self.units))
#             nn.init.xavier_uniform_(self.bias.data, gain=1.414)

#     def forward(self, inputs):

#         features, node, neighbours = inputs

#         node_feat = nn.embedding_lookup(features, node)
#         neigh_feat = nn.embedding_lookup(features, neighbours)

#         dims = torch.shape(neigh_feat)
#         batch_size = dims[0]
#         num_neighbors = dims[1]
#         h_reshaped = torch.reshape(
#             neigh_feat, (batch_size * num_neighbors, self.input_dim))

#         for l in self.dense_layers:
#             h_reshaped = l(h_reshaped)
#         neigh_feat = torch.reshape(
#             h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))

#         if self.pooling == "meanpooling":
#             neigh_feat = torch.reduce_mean(neigh_feat, axis=1, keep_dims=False)
#         else:
#             neigh_feat = torch.reduce_max(neigh_feat, axis=1)

#         output = torch.concat(
#             [torch.squeeze(node_feat, axis=1), neigh_feat], axis=-1)

#         output = torch.matmul(output, self.neigh_weights)
#         if self.use_bias:
#             output += self.bias
#         if self.activation:
#             output = self.activation(output)

#         return output

#     def get_config(self):
#         config = {'output_dim': self.output_dim,
#                   'concat': self.concat
#                   }

#         base_config = super(PoolingAggregator, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

# class GraphSAGE(nn.Module):
#     def __init__(self,
#                  feature_dim, #输入特征维度
#                  neighbor_num, #采样邻居个数
#                  n_hidden, #中间层特征维度
#                  n_classes, #类别数
#                  use_bias=True,
#                  activation=nn.relu,
#                  aggregator_type='mean', #聚合函数选择
#                  dropout_rate=0.0,
#                  l2_reg=0):
#         super(GraphSAGE, self).__init__()
#         if aggregator_type == 'mean':
#             self.aggregator = MeanAggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
#                        dropout_rate=dropout_rate, neigh_max=2, aggregator=aggregator_type)
#         else:
#             self.aggregator = PoolingAggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
#                        dropout_rate=dropout_rate, neigh_max=2, aggregator=aggregator_type)

#     def forward(input, adj, neigh_number):
#         '''
#         neigh_number = [2,2] 也就是选择邻居节点的个数
#         '''
#         sample_neigh, sample_neigh_len = sample_neighs(
#             adj, adj.shape[0], neigh_number[0], self_loop=False)

#         x = self.aggregator([sample_neigh])


# def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):  # 抽样邻居节点
#     _sample = np.random.choice
#     neighs = [list(G[int(node)]) for node in nodes]  # nodes里每个节点的邻居
#     if sample_num:
#         if self_loop:
#             sample_num -= 1

#         samp_neighs = [
#             list(_sample(neigh, sample_num, replace=False)) if len(neigh) >= sample_num else list(
#                 _sample(neigh, sample_num, replace=True)) for neigh in neighs]  # 采样邻居
#         if self_loop:
#             samp_neighs = [
#                 samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # gcn邻居要加上自己

#         if shuffle:
#             samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
#     else:
#         samp_neighs = neighs
#     return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))


import sys
import os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        #self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
        self.layer = nn.Sequential(
            nn.Linear(emb_size, num_classes)
            # nn.ReLU()
        )
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.layer(embeds), 1)
        return logists

# class Classification(nn.Module):

# 	def __init__(self, emb_size, num_classes):
# 		super(Classification, self).__init__()

# 		self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
# 		self.init_params()

# 	def init_params(self):
# 		for param in self.parameters():
# 			nn.init.xavier_uniform_(param)

# 	def forward(self, embeds):
# 		logists = torch.log_softmax(torch.mm(embeds,self.weight), 1)
# 		return logists


class UnsupervisedLoss(object):
    """docstring for UnsupervisedLoss"""

    def __init__(self, adj_lists, train_nodes, device):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 6
        self.WALK_LEN = 1
        self.N_WALK_LEN = 5
        self.MARGIN = 3
        self.adj_lists = adj_lists
        self.train_nodes = train_nodes
        self.device = device

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i]
                             for i in range(len(nodes))]
        node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(
                embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q * \
                torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            # print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(
                embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            # print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1, -1))

        loss = torch.mean(torch.cat(nodes_score, 0))

        return loss

    def get_loss_margin(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i]
                             for i in range(len(nodes))]
        node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(
                embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(
                embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

            nodes_score.append(torch.max(torch.tensor(0.0).to(
                self.device), neg_score-pos_score+self.MARGIN).view(1, -1))
            # nodes_score.append((-pos_score - neg_score).view(1,-1))

        loss = torch.mean(torch.cat(nodes_score, 0), 0)

        # loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))

        return loss

    def extend_nodes(self, nodes, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        # print(self.positive_pairs)
        self.get_negtive_nodes(nodes, num_neg)
        # print(self.negtive_pairs)
        self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([
                                       i for x in self.negtive_pairs for i in x]))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes):
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes, num_neg):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(
                far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node)
                                      for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [
                (node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = self.adj_lists[int(curr_node)]
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.train_nodes:
                        self.positive_pairs.append((node, next_node))
                        cur_pairs.append((node, next_node))
                    curr_node = next_node

            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(
            out_size, self.input_size if self.gcn else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, adj_lists, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        self.adj_lists = adj_lists

        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer'+str(index),
                    SageLayer(layer_size, out_size, gcn=self.gcn))

        self.fc = nn.Linear(out_size*5, 2)

    def forward(self, input, adj):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        nodes_batch = [i for i in range(len(adj))]
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes,num_sample=2)
            nodes_batch_layers.insert(
                0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = input
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        pre_hidden_embs = pre_hidden_embs.contiguous().view(1, -1)
        out_put = self.fc(pre_hidden_embs)
        return nb, pre_hidden_embs, F.log_softmax(out_put, dim=1)

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(
                to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]])
                       for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i])
                     for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]]))
                           for i in range(len(samp_neighs))]
        # self.dc.logger.info('2')
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n]
                          for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs))
                       for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # self.dc.logger.info('4')

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            # print(mask)
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            # self.dc.logger.info('5')
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        # self.dc.logger.info('6')

        return aggregate_feats
