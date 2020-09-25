# -*- coding: UTF-8 -*-
"""
@Author  : Kuang Zhengze
@Time    : 2020/9/16 21:43
@File    : main.py
"""
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE
from torch_geometric.utils import train_test_split_edges

import args
from model import Encoder, VEncoder, get_edge_acc

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = None
if args.dataset.lower() == 'Cora'.lower():
    dataset = Planetoid(root='tmp', name='Cora')
    print("use dataset: Cora")
elif args.dataset.lower() == 'CiteSeer'.lower():
    dataset = Planetoid(root='tmp', name='CiteSeer')
    print("use dataset: CiteSeer")
elif args.dataset.lower() == 'PubMed'.lower():
    dataset = Planetoid(root='tmp', name='PubMed')
    print("use dataset: PubMed")
data = dataset[0]

enhanced_data = train_test_split_edges(data.clone(),val_ratio=0.1,test_ratio=0.2)

train_data = Data(x=enhanced_data.x,edge_index=enhanced_data['train_pos_edge_index']).to(DEVICE)
target_data = data.to(DEVICE)


if args.model is 'VGAE':
    model = VGAE(encoder=VEncoder(data['x'].shape[1])).to(DEVICE)
else:
    model = GAE(encoder=Encoder(data['x'].shape[1])).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)


def model_train():
    print("========Start training========")
    for epoch in range(args.num_epoch):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data)
        recon_loss = model.recon_loss(z, target_data['edge_index'])
        if args.model is 'VGAE':
            recon_loss += model.kl_loss()/data['x'].shape[0]
        recon_loss.backward()
        optimizer.step()
        if (epoch) % 5 == 0:
            model.eval()
            val_roc,val_ap = model.test(z,enhanced_data['val_pos_edge_index'],enhanced_data['val_neg_edge_index'])
            acc,_,_,_,_ = get_edge_acc(model.decoder.forward_all(z),target_data)
            print("Epoch {:0>4d} : train_loss={:.5f}, val_roc={:.5f}, val_ap={:.5f}, acc={:.7f}"
                  .format(epoch,recon_loss, val_roc, val_ap,acc))
    print("========   Finished   ========")

def model_test():
    model.eval()
    z = model.encode(data.to(DEVICE))
    test_roc, test_ap = model.test(z, enhanced_data['val_pos_edge_index'], enhanced_data['val_neg_edge_index'])
    acc, tn, fp, fn, tp = get_edge_acc(model.decoder.forward_all(z),target_data)
    print("Test result: test_roc={:.5f}, test_ap={:.5f}, acc={:.7f}".format(test_roc, test_ap, acc))

    print([[tn,fp],[fn,tp]])

if __name__ == '__main__':
    model_train()
    model_test()



