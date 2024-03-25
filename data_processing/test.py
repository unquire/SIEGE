# import pandas as pd
#
# # 读取第一个 CSV 文件
# df1 = pd.read_csv('../data/origin_data.csv')
#
# # 读取第二个 CSV 文件
# df2 = pd.read_csv('../data/PhishingAddress.csv')
#
# # 提取两个文件中的地址列
# addresses1_1 = df1['from'].tolist()  # 替换 'address_column_name' 为实际的地址列名
# addresses1_2 = df1['to'].tolist()
# addresses2 = df2['address'].tolist()  # 替换 'address_column_name' 为实际的地址列名
#
# # 统计地址在第二个 CSV 文件中出现的次数
# count_1 = sum(address in addresses2 for address in addresses1_1)
# count_2 = sum(address in addresses2 for address in addresses1_2)
#
# print(f"Address appears {count_1+count_2} times in the second CSV file.")

# import os
# import pandas as pd
#
# # 指定文件夹路径
# folder_path = '/root/YMT/NeiMengGu/data/钓鱼一阶节点'
#
# # 获取文件夹中的所有文件名
# file_names = os.listdir(folder_path)
# file_names = [file.split('.')[0] for file in file_names]
# # 创建 DataFrame
# df1 = pd.DataFrame({'address': file_names,'label':1})
# print("第一个df大小是:",df1.shape)
# df2 = pd.read_csv('../data/PhishingAddress.csv')
# print("第二个df大小是:",df2.shape)
# df_concat = pd.concat([df1, df2])
# print("合并df大小是:",df_concat.shape)
# df_concat = df_concat.drop_duplicates()
# print(df_concat.shape)
# # 保存 DataFrame 到 CSV 文件
# df_concat.to_csv('../data/PhishingAddress.csv', index=False)

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# import numpy as np
#
# # 生成一些多维样本数据
# np.random.seed(0)
# X = np.random.rand(100, 5)  # 100个样本，每个样本有5个特征
# print(X[:2])
# # 初始化K均值模型并拟合数据
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
#
# # 获取簇中心点和每个样本所属的簇标签
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
#
# print("Cluster Centers:\n", centroids)
# print("\nCluster Labels:\n", labels)
#
# # 将聚类标签添加为新的特征
# X_with_labels = np.hstack((X, labels.reshape(-1, 1)))
#
# print("Original Feature Matrix with Cluster Labels:")
# print(X_with_labels)

# 补救地址重复的代码
# import pandas as pd
# import torch
# # 假设你已经有一个 DataFrame df 和一个标签列表 labels
# # df 是原始 DataFrame
# # labels 是新产生的标签列表，长度与 df 中的行数相同
# file_name = "origin_split_1"
# pd.set_option('display.max_columns', None)
# train_ids = pd.read_csv("../data/{}/address.csv".format(file_name))
# train_dup = train_ids
# print(train_ids)
# train_ids = train_ids[['address']].copy()
# print("train_ids:",train_ids.shape)
# train_ids = train_ids.drop_duplicates(keep=False).reset_index(drop=True)
# print("train_ids1:",train_ids.shape)
# train = train_ids.merge(train_dup,how='left',left_on='address',right_on='address')
# print(train.head(1))
# print(train.shape)
# df_contract = train[train['IsContract'] == 1]
# print("智能合约地址数量:", df_contract.shape)
# train.to_csv("../data/{}/address.csv".format(file_name), index=False)
# df_contract.to_csv("../data/{}/contractAddress.csv".format(file_name), index=False)
#
# feature = pd.read_csv("../data/{}/features.csv".format(file_name))
# feature = train_ids.merge(feature,how='left',left_on='address',right_on='address')
# feature.to_csv("../data/{}/features.csv".format(file_name),index=False)
# address_to_index = {row['address']: index for index, row in feature.iterrows()}
# torch.save(address_to_index, '../data/{}/address_to_index.pt'.format(file_name))
# features = feature.drop(['address'], axis=1)
# torch.save(features.values, '../data/{}/features_matrix.pt'.format(file_name))
# print("程序已完成！！！！！！！")
# print("程序结束时间:")

# 计算train_tx代码
# import pandas as pd
# import torch
# df = pd.read_csv("../data/origin_split_1/transaction.csv")
# df_from_to = df[['from','to']].copy()
# print(df_from_to.head())
# df_from_to.to_csv("../graph_data/raw/train_tx.csv",index=False)
# # df = pd.read_csv("../graph_data/raw/train_tx.csv")
# # print(df.head())
#
# df_contract = pd.read_csv("../data/origin_split_1/contractAddress_add_cluter.csv")
# address_to_index = torch.load("../data/origin_split_1/address_to_index.pt")
# df_contract[['address']] = df_contract[['address']].applymap(lambda addr: address_to_index[addr])

# 构建第一个address_to_index.pt 与第二个address_to_index.pt的对应关系
# import torch
#
# # 加载第一个地址转索引的 pt 文件
# address_to_index_1 = torch.load('../data/origin_split_1/address_to_index.pt')
#
# # 加载第二个地址转索引的 pt 文件
# address_to_index_2 = torch.load('../data/origin_split_2/address_to_index.pt')
#
# # 创建一个新的字典用于存储最终的地址转地址索引映射
# address_to_address = {}
#
# # 遍历第一个地址转索引的字典
# for address, index_1 in address_to_index_1.items():
#     # 如果第二个字典中存在相同的地址
#     if address in address_to_index_2:
#         # 获取第二个地址对应的索引
#         index_2 = address_to_index_2[address]
#         # 将第一个地址对应的索引映射到第二个地址对应的索引
#         address_to_address[index_1] = index_2
#
# print("Size:",len(address_to_address))
# # 输出前五个索引对应的映射
# for i, (index_1, index_2) in enumerate(address_to_address.items()):
#     print(f"Index 1: {index_1}, Index 2: {index_2}")
#     if i == 4:
#         break  # 输出前五个条目
# torch.save(address_to_address, '../data/origin_split_1/address1_to_address2.pt')

# 创建智能合约地址转换完的文件
# import pandas as pd
# import torch
# address_to_index = torch.load('../data/origin_split_1/address_to_index.pt')
# df = pd.read_csv("../data/origin_split_1/contractAddress_add_cluter.csv")
# print(df.head())
# df.drop(df[(~df['address'].isin(address_to_index))].index,inplace=True)
# df[['address']] = df[['address']].applymap(lambda addr: address_to_index[addr])
# print("length:",df.shape)
# print(df.head())
# df.to_csv("../data/origin_split_1/contractAddress_add_cluter_changeAddress.csv", index=False)

from tqdm import tqdm
import time

from tqdm import tqdm
import time

# from tqdm import tqdm
# import time
#
# data1 = range(10)
# data2 = range(5)
#
# # 禁用输出缓冲
# tqdm.monitor_interval = 0
#
# with tqdm(total=len(data1), desc="Processing data", unit="item") as pbar:
#     for item1 in data1:
#         # 手动更新外部循环的进度条
#         pbar.update(1)
#
#         # 创建内部循环的进度条
#         pbar2 = tqdm(total=len(data2), desc="Processing data 2", unit="item1", position=1)
#         for item2 in data2:
#             time.sleep(0.1)
#             # 更新内部循环的进度条
#             pbar2.update(1)
#         # 关闭内部循环的进度条
#        pbar2.close()

import pandas as pd
# import torch
# df = torch.load("../data/origin_split_1/address_to_index.pt")
# for address, index in list(df.items())[:5]:
#     print(f"Address: {address}, Index: {index}")
import pandas as pd
import torch
# 假设 csv_file 是你的 CSV 文件路径
# df = pd.read_csv("../data/origin_split_1/test.csv")
# print(df.shape)
# # 创建 DataFrame
# data = pd.DataFrame({'edge_type': range(df.shape[0])})
# x = torch.load('../data/origin_split_1/features_matrix.pt')
# original_shape = x.shape
# print(x.shape)
# print(type(x))
# # 计算扩展后的列维度
# expanded_shape = (original_shape[0], original_shape[1] * 2)
# # 创建一个具有扩展形状的零张量
# expanded_x = torch.zeros(expanded_shape)
#
# # 将原始张量的值复制到扩展后的张量中
# expanded_x[:, :original_shape[1]] = torch.tensor(x)
# print(expanded_x.shape)
# # 输出 DataFrame
# print(data)
# print(data.shape)
# print(expanded_x.shape)
# print("第一个元素的值:", expanded_x[:5,:])
# df = pd.read_csv("../data/origin_split_1/features.csv")
# pd.set_option('display.max_columns', None)
# print(df.head(1))

# 创建graph的相关标记
# 特征数据
# import torch
# import pandas as pd
# import numpy as np
# address1_to_address2 = torch.load("../data/origin_split_1/address1_to_address2.pt")
# # 加载智能合约标签数据
# contractAddress = pd.read_csv("../data/origin_split_1/contractAddress_add_cluter_changeAddress.csv")
# nextfeature = torch.load("../data/origin_split_2/features_matrix.pt")
# x = torch.load('../data/origin_split_1/features_matrix.pt')
# print("type(x)=",type(x))
# create_feature = pd.DataFrame({'address': np.arange(len(x))})
# print(create_feature.head(5))
# create_feature = create_feature.merge(contractAddress,how='left',left_on='address',right_on='address')
# create_feature = create_feature.fillna(6)
# print(create_feature.head(5))
# 创建新的特征矩阵
# new_features_matrix = []
# # 遍历第一个特征矩阵的行
# for row_index in range(x.shape[0]):
#     # 获取与第一个特征矩阵中当前行对应的第二个特征矩阵的行数
#     corresponding_row_index = address1_to_address2.get(row_index)
#
#     if corresponding_row_index is not None:
#         # 根据映射关系取出第二个特征矩阵中的相应行
#         corresponding_row = nextfeature[corresponding_row_index]
#     else:
#         # 如果没有找到对应的映射关系，用全是0的数组补充
#         corresponding_row = np.zeros(nextfeature.shape[1])  # 或者使用 torch.zeros() 也可以
#     # 将新行添加到新的特征矩阵中
#     new_features_matrix.append(corresponding_row)
# print(new_features_matrix[:5])

import torch

# 假设你有两个张量 tensor1 和 tensor2
# import torch
#
# # 假设你有两个张量 tensor1 和 tensor2
# tensor1 = torch.tensor([[0, 1, 0],
#                         [0, 0, 0],
#                         [1, 0, 0],
#                         [0, 0, 1]])
#
# tensor2 = torch.tensor([[0.1, 0.2],
#                         [0.3, 0.4],
#                         [0.5, 0.6],
#                         [0.7, 0.8]])
#
# # 计算 tensor1 中每列是否全为 0
# condition = torch.sum(tensor1, dim=1) != 0
#
# # 根据条件过滤 tensor1 和 tensor2 中的列
# filtered_tensor1 = tensor1[condition]
# filtered_tensor2 = tensor2[condition]
#
# # 打印结果
# print("Filtered tensor1:")
# print(filtered_tensor1)
#
# print("Filtered tensor2:")
# print(filtered_tensor2)
#
# # 获取 filtered_tensor1 的大小
# filtered_tensor1_size = filtered_tensor1.size(0)
#
# print("Size of filtered_tensor1:", filtered_tensor1_size)

# import torch
# import torch.nn.functional as F
#
# # 假设 x 是模型的输出，每行代表一个样本的预测分数
# x = torch.tensor([[1.0, 2.0, 3.0],
#                   [4.0, 5.0, 6.0]])
#
# # 在最后一个维度上应用 log softmax 操作
# log_probs = F.log_softmax(x, dim=-1)
#
# print("Log probabilities after log softmax:")
# print(log_probs)
import sys
sys.path.append('/home/YMT/SIEGE/')
# sys.path.append('/home/YMT/SIEGE/model')
print(sys.path)