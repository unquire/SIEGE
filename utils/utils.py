import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import List, Optional, Union
from torch import Tensor
from torch.utils.data import Dataset as utils_Dataset
from torch_geometric.data import Dataset, Data
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cos
import json
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
import os
import shutil


def prepare_folder(path, model_name):
    model_dir = f'{path}/model_files/{model_name}/'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def graph_process(raw_paths):
    # # 开始处理交易数据
    TX_COLUMNS = ['from', 'to']
    trans = pd.read_csv(raw_paths[0], usecols=TX_COLUMNS)
    # trans['timestamp'] = pd.to_datetime(trans['timestamp']).view(int)//1e9  # 这里还没有用到时序数据
    address_to_index = torch.load('../data/origin_split_1/address_to_index.pt')  # address_to_index可能是有address的行
    # 删除地址不在index里的行，因为这些账户没有特征
    trans.drop(
        trans[(~trans['from'].isin(address_to_index)) | (~trans['to'].isin(address_to_index))].index,
        inplace=True)
    trans[['from', 'to']] = trans[['from', 'to']].applymap(
        lambda addr: address_to_index[addr])  # 将from_address和to_address转成index
    # -------扩大特征数据维度的，使得维度是以前的二倍
    x = torch.load('../data/origin_split_1/features_matrix.pt')
    # x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    # 开始处理特征维度
    original_shape = x.shape
    # 计算扩展后的列维度
    expanded_shape = (original_shape[0], original_shape[1] * 2)
    # 创建一个具有扩展形状的零张量
    expanded_x = torch.zeros(expanded_shape)
    # 将原始张量的值复制到扩展后的张量中
    expanded_x[:, :original_shape[1]] = torch.tensor(x)
    data_index = pd.DataFrame({'node_index': range(x.shape[0])})

    address1_to_address2 = torch.load("../data/origin_split_1/address1_to_address2.pt")
    # 加载智能合约标签数据
    contractAddress = pd.read_csv("../data/origin_split_1/contractAddress_add_cluter_changeAddress.csv")
    nextfeature = torch.load("../data/origin_split_2/features_matrix.pt")

    x = torch.load('../data/origin_split_1/features_matrix.pt')
    # print("type(x)=",type(x))
    create_feature = pd.DataFrame({'address': np.arange(len(x))})
    # print(create_feature.head(5))
    create_feature = create_feature.merge(contractAddress,how='left',left_on='address',right_on='address')
    create_feature = create_feature.fillna(6)
    # print(create_feature.head(5))
    # 创建新的特征矩阵
    new_features_matrix = []
    # 遍历第一个特征矩阵的行
    for row_index in range(x.shape[0]):
        # 获取与第一个特征矩阵中当前行对应的第二个特征矩阵的行数
        corresponding_row_index = address1_to_address2.get(row_index)

        if corresponding_row_index is not None:
            # 根据映射关系取出第二个特征矩阵中的相应行
            corresponding_row = nextfeature[corresponding_row_index]
        else:
            # 如果没有找到对应的映射关系，用全是0的数组补充
            corresponding_row = np.zeros(nextfeature.shape[1])  # 或者使用 torch.zeros() 也可以
        # 将新行添加到新的特征矩阵中
        new_features_matrix.append(corresponding_row)
    data = Data(x=expanded_x.float(),
                edge_index=torch.tensor([trans['from'].values, trans['to'].values]).to(torch.int64),
                # edge_index_reconn=torch.tensor([reconn_src, reconn_dst]).to(torch.int64),
                # node_index=torch.tensor(data_index['node_index'].values).to(torch.int64),
                contract_cluter=torch.tensor(create_feature['Label'].values).to(torch.int64),# 合约地址分组，不是合约地址为标记为6。
                next_features = torch.tensor(new_features_matrix).float()  # 关联地址的特征，用于计算接口任务三。
                # y=torch.tensor(y.values).to(torch.long), num_nodes=len(x)
    )
    # print(data.edge_index)
    return data


class GNNDataset(Dataset):
    # 继承了父类中Dataset类初始化方法
    def __init__(self, root, transform=None, pre_transform=None):
        super(GNNDataset, self).__init__(root, transform, pre_transform)

    # 原始文件位置
    @property
    def raw_file_names(self):
        return ['train_tx.csv']

    # 文件保存位置
    @property
    def processed_file_names(self):
        return 'graph.pt'

    def download(self):
        pass

    def process(self):
        data = graph_process(self.raw_paths)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.7,
                 gamma=3,
                 reduction='mean', ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                                F.softplus(logits, -1, 50),
                                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                                  -logits + F.softplus(logits, -1, 50),
                                  -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class MyDataset(utils_Dataset):
    def __init__(self, walks, attrs, labels):
        self.walks = walks
        self.attrs = attrs
        self.labels = labels
        self.data_ = [(walk_item, attr_item, label_item) for walk_item, attr_item, label_item in
                      zip(walks, attrs, labels)]

    def __getitem__(self, index):
        return self.data_[index]

    def __len__(self):
        return len(self.data_)


class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(
            self,
            dataset: Union[Tensor],
            input_nodes: Optional[Tensor] = None,
            num_samples: Optional[int] = None,
    ):
        y = dataset.view(-1)
        y = y[input_nodes] if input_nodes is not None else y

        assert y.dtype == torch.long  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1. / y.bincount()
        weight = class_weight[y]

        return super().__init__(weight, num_samples, replacement=True)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, loss, model):

        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            pass
            # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.loss_min = loss
