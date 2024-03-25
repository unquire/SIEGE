import pandas as pd
from tqdm import tqdm
# 读取 CSV 文件
df = pd.read_csv('/home/LZM/data/data0-13/5000000to5999999_BlockTransaction.csv')
pd.set_option('display.max_columns', None)
# 删除符合条件的行
df = df.drop(df[df['blockNumber'] < 5078351].index).reset_index(drop=True)
df = df.drop(df[df['blockNumber'] > 5324013].index).reset_index(drop=True)
df = df.drop(df[(df['value'] == '0') & (df['fromIsContract'] == '0') & (df['toIsContract'] == '0')].index).reset_index(
    drop=True)
df = df.drop(df[df['isError'] == 'True'].index).reset_index(drop=True)
# 显示数据框的内容
print(df.head())
print(df.shape)
df_from = df[['from', 'fromIsContract']]
df_from.rename(columns={'from': 'address', 'fromIsContract': 'IsContract'},inplace=True)
# print("df_to:",df_from)
df_to = df[['to', 'toIsContract']]
df_to.rename(columns={'to': 'address', 'toIsContract': 'IsContract'}, inplace=True)
# print("df_to:",df_to)
df_node = pd.concat([df_from, df_to]).drop_duplicates().reset_index(drop=True)

# 过滤既是合约地址，又是普通地址的地址
train_ids = df_node[['address']].copy()
train_ids = train_ids.drop_duplicates(keep=False).reset_index(drop=True)
print("train_ids1:",train_ids.shape)
train = train_ids.merge(df_node,how='left',left_on='address',right_on='address')
df_node = train

df_contract = df_node[df_node['IsContract'] == 1]

print("节点数为:", df_node.shape)
print(df_node.head())
print("智能合约地址数为:", df_contract.shape)
print(df_contract.head())
df.to_csv('../data/origin_split_2/transaction.csv', index=False)
df_node.to_csv('../data/origin_split_2/address.csv', index=False)
df_contract.to_csv('../data/origin_split_2/contractAddress.csv', index=False)

#------------------------去重------------------------------------
# df1_node = pd.read_csv('../data/origin_split_1/address.csv')
# df1_contract = pd.read_csv('../data/origin_split_1/contractAddress.csv')
# df_node_concat = pd.concat([df1_node,df_node]).drop_duplicates().reset_index(drop=True)
# df_contract_concat = pd.concat([df1_contract,df_contract]).drop_duplicates().reset_index(drop=True)

# df.to_csv('../data/origin_split_1/transaction.csv', mode='a', header=False, index=False)
# df_node_concat.to_csv('../data/origin_split_1/address.csv', index=False)
# df_contract_concat.to_csv('../data/origin_split_1/contractAddress.csv', index=False)
print("程序已完成！！！！")
