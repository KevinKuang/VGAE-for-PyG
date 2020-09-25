# -*- coding: UTF-8 -*-
"""
@Author  : Kuang Zhengze
@Time    : 2020/9/16 21:34
@File    : model.py
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, remove_self_loops, add_self_loops

import args


class Encoder(torch.nn.Module):
    def __init__(self,node_features):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(node_features, args.hidden1_dim)
        self.conv2 = GCNConv(args.hidden1_dim, args.hidden2_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        return x


class VEncoder(torch.nn.Module):

    def __init__(self,node_features):
        super(VEncoder, self).__init__()
        self.common_conv1 = GCNConv(node_features, args.hidden1_dim)
        self.mean_conv2 = GCNConv(args.hidden1_dim, args.hidden2_dim)
        self.logstd_conv2 = GCNConv(args.hidden1_dim, args.hidden2_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.common_conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        mean = self.mean_conv2(x, edge_index)

        logstd = self.logstd_conv2(x, edge_index)


        return mean, logstd

def get_edge_acc(pred_adj,target_data):
    # pred_adj.cpu()
    # pred_adj = torch.pow(pred_adj,2)
    # print(pred_adj.shape)

    pred_adj[pred_adj >= args.threshold] = 1
    pred_adj[pred_adj < args.threshold] = 0
    edge_index,_ = add_self_loops(target_data['edge_index'])
    target_adj = to_dense_adj(edge_index)[0]
    tn, fp, fn, tp = confusion_matrix(target_adj.view(-1).cpu().detach().numpy(), pred_adj.view(-1).cpu().detach().numpy()).ravel()
    # print(target_adj.shape)
    # check_adj = pred_adj + target_adj
    # check_adj[check_adj >= 1] = 1
    # check_adj[check_adj < 1] = 0

    return (tn+tp)/(tn+fp+fn+tp),tn, fp, fn, tp

