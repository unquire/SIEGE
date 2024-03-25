# 构建特征矩阵
import os
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from datetime import datetime


COLUMNS = ['address', 'IsContract']
train_ids = pd.read_csv(f'../data/origin_split_2/address.csv', usecols=COLUMNS)
print("train_ids:",train_ids.shape)
TX_DTYPES = {'value': 'float', 'from': 'object', 'to': 'object', 'timestamp': 'int'}
IN_COLUMNS = ['value', 'from', 'to', 'timestamp']

trans = pd.read_csv('../data/origin_split_2/transaction.csv', usecols=IN_COLUMNS, dtype=TX_DTYPES)
result = pow(10, 18)
trans['value'] = trans['value'] / float(result)
trans = trans.drop_duplicates().reset_index(drop=True)
pd.set_option('display.max_columns', None)
# 计算多重边边出现的比例
# data = {'from': [1, 1, 4, 4],
#         'to': [2, 2, 3, 3],
#         'timestamp':[2,3,4,5]}
# df = pd.DataFrame(data)

mult_coming = trans.groupby(['from', 'to']).agg({'timestamp': ['count']}).reset_index()
# print(mult_coming)
mult_coming.columns = ["_".join(filter(None, name)) for name in mult_coming.columns.to_flat_index()]  # 将多行列名替换为单行列名
mult_coming = mult_coming.rename(columns={'timestamp_count': 'timestamp_count_mult'})
mult_coming = mult_coming[mult_coming['timestamp_count_mult'] > 1].reset_index(drop=True)
mult_coming_in = mult_coming.groupby('to').agg({'from': ['count']}).reset_index()
# print("mult:", mult_coming['from'].dtype)
mult_coming_in.columns = ["_".join(filter(None, name)) for name in mult_coming_in.columns.to_flat_index()]  # 将多行列名替换为单行列名

mult_coming_out = mult_coming.groupby('from').agg({'to': ['count']}).reset_index()
mult_coming_out.columns = ["_".join(filter(None, name)) for name in mult_coming_out.columns.to_flat_index()]  # 将多行列名替换为单行列名
# print(mult_coming_in)
# print(mult_coming_out)
# # 计算当前地址的入度

# incoming = incoming.dropna(subset=['address'])
incoming_agg = trans.groupby('to').agg({'value': ['sum', 'mean'],
                                        'timestamp': ['min', 'max', 'median', 'std', 'mean', 'count'],  # 时间跨度
                                        'from': ['nunique']
                                        }).reset_index()
incoming_agg.columns = ["_".join(filter(None, name)) for name in incoming_agg.columns.to_flat_index()]  # 将多行列名替换为单行列名
incoming_agg = incoming_agg.rename(columns={  # 'value_eth_min': 'min_value_eth_in',
    'to': 'address',
    'value_sum': 'sum_value_in',
    'value_mean': 'mean_value_in',
    'timestamp_min': 'timestamp_min_in',
    'timestamp_max': 'timestamp_max_in',
    'timestamp_median': 'timestamp_median_in',
    'timestamp_std': 'timestamp_std_in',
    'timestamp_mean': 'timestamp_mean_in',
    'timestamp_count': 'tx_number_in',
    'from_nunique': 'in_degree'
})
pd.set_option('display.max_columns', None)
# print("incoming_agg:",incoming_agg.head())

# 计算当前地址的出度
# incoming = incoming.dropna(subset=['address'])
outcoming_agg = trans.groupby('from').agg({'value': ['sum', 'mean'],
                                           # median 中值
                                           # 2023.9.27  添加timestamp并注释掉block_number
                                           'timestamp': ['min', 'max', 'median', 'std', 'mean', 'count'],  # 时间跨度
                                           # 'block_number': ['std', 'max', 'min']
                                           'to': ['nunique']
                                           }).reset_index()
outcoming_agg.columns = ["_".join(filter(None, name)) for name in outcoming_agg.columns.to_flat_index()]  # 将多行列名替换为单行列名
outcoming_agg = outcoming_agg.rename(columns={  # 'value_eth_min': 'min_value_eth_in',
    'from': 'address',
    'value_sum': 'sum_value_out',
    'value_mean': 'mean_value_out',
    'timestamp_min': 'timestamp_min_out',
    'timestamp_max': 'timestamp_max_out',
    'timestamp_median': 'timestamp_median_out',
    'timestamp_std': 'timestamp_std_out',
    'timestamp_mean': 'timestamp_mean_out',
    'timestamp_count': 'tx_number_out',
    'to_nunique': 'out_degree'
})
# print("outcoming_agg:",outcoming_agg.head())

feature = train_ids.merge(incoming_agg, how='left', left_on='address', right_on='address')
feature = feature.merge(outcoming_agg,how='left',left_on='address',right_on='address')
feature = feature.merge(mult_coming_in, how='left', left_on='address', right_on='to')
feature = feature.merge(mult_coming_out, how='left', left_on='address', right_on='from')
feature = feature.fillna(0)

feature['sum_value'] = feature['sum_value_in'] + feature['sum_value_out']

feature['tx_span'] = feature.apply(
    lambda row: max(row['timestamp_max_out'], row['timestamp_max_in']) - min(row['timestamp_min_out'],
                                                                             row['timestamp_min_in']), axis=1)
feature['tx_span_in'] = feature['timestamp_max_in'] - feature['timestamp_min_in']
feature['tx_span_out'] = feature['timestamp_max_out'] - feature['timestamp_min_out']
feature['tx_freq_in'] = feature['tx_span_in'] / feature['tx_number_in']
feature['tx_freq_out'] = feature['tx_span_out'] / feature['tx_number_out']
feature['multi_edge_in'] = feature['from_count'] / feature['in_degree']
feature['multi_edge_out'] = feature['to_count'] / feature['out_degree']
feature = feature.drop(['from_count', 'to_count', 'from', 'to'], axis=1)
feature = feature.fillna(0)
print("feature:", feature.head(2))
print(feature.shape)

# # # 生成特征矩阵
feature.to_csv("../data/origin_split_2/features.csv", index=False)
address_to_index = {row['address']: index for index, row in feature.iterrows()}
torch.save(address_to_index, '../data/origin_split_2/address_to_index.pt')
features = feature.drop(['address'], axis=1)
torch.save(features.values, '../data/origin_split_2/features_matrix.pt')
print("程序已完成！！！！！！！")
print("程序结束时间:")
