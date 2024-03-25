from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

class SAGE_NeighSampler(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , out_class
                 , dropout
                 , batchnorm=True):
        super(SAGE_NeighSampler, self).__init__()

        # if num_layers == 1:
        #     self.convs = torch.nn.ModuleList()
        #     self.convs.append(SAGEConv(in_channels, out_channels))
        #     self.bns = torch.nn.ModuleList()
        #     self.batchnorm = batchnorm
        #     self.num_layers = num_layers
        # else:
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.out_class = out_class
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.decoder1 = torch.nn.Linear(out_channels, in_channels)  # 任务一是空间接口任务
        self.decoder2 = torch.nn.Linear(out_channels, out_class)  # 任务二是智能合约分类接口任务
        self.decoder3 = torch.nn.Linear(out_channels, out_channels)  # 任务三是相同地址分类的接口任务
        self.dropout = dropout

        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        # Reconstruction for task 1
        recon_x1 = self.decoder1(x)

        # Reconstruction for task 2
        recon_x2 = self.decoder2(x)  # Use all nodes

        # Reconstruction for task 3 (using selected nodes)
        recon_x3 = self.decoder3(x)  # Use all nodes

        return recon_x1, recon_x2, recon_x3
        # return x.log_softmax(dim=-1)
    def self_supervised_loss(self,x,recon_x1,recon_x2,recon_x3,next_feature,contract_cluter):
        # 计算第一个损失函数，空间损失函数
        recon_x1_size = float(recon_x1.size(0))
        x_size = x.size(0)
        # print("Size of loss1:", recon_x1_size, " ", x_size)
        # print("recon_x1:",recon_x1)
        # print("x:",x)
        recon_loss1 = F.mse_loss(recon_x1, x)/recon_x1_size
        # print("loss1:",recon_loss1)
        # 计算第二个损失函数，智能合约损失函数
        condition = contract_cluter != 6
        recon_x2 = recon_x2[condition]
        contract_cluter = contract_cluter[condition]
        recon_x2_size = float(recon_x2.size(0))
        if recon_x2_size != 0:
            recon_x2 = F.log_softmax(recon_x2, dim=-1)
            recon_loss2 = F.nll_loss(recon_x2, contract_cluter)/recon_x2_size
            # print("recon_loss2:", recon_loss2)
        else:
            recon_loss2 = 0
        contract_cluter_size = contract_cluter.size(0)
        # print("recon_x2:", recon_x2)
        # print("contract_cluter:", contract_cluter)
        # print("Size of loss2:", recon_x2_size, " ", contract_cluter_size)
        # print("loss2:", recon_loss2)
        # 计算第三个损失函数，时间损失函数
        # 过滤符合条件的值
        condition = torch.sum(next_feature, dim=1) != 0
        # 根据条件过滤 recon_x3 和 next_feature 中的列
        recon_x3 = recon_x3[condition]
        next_feature = next_feature[condition]
        recon_x3_size = float(recon_x3.size(0))
        next_feature_size = next_feature.size(0)
        if recon_x3_size != 0:
            recon_loss3 = F.mse_loss(recon_x3, next_feature)/recon_x3_size
        else:
            recon_loss3 = 0
        # 获取 filtered_tensor1 的大小
        # print("recon_x3:", recon_x3)
        # print("next_feature:", next_feature)
        # print("Size of loss3:", recon_x3_size," ",next_feature_size)
        # print("loss3:", recon_loss3)
        loss = recon_loss1 + recon_loss2 + recon_loss3
        # print("loss:",loss)
        return loss

