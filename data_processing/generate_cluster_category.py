from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
# 读入初始化数据
file_name = "origin_split_1"
df_features = pd.read_csv("../data/{}/features.csv".format(file_name))
df_contract = pd.read_csv("../data/{}/contractAddress.csv".format(file_name))
print("df_contract:",df_contract.shape)
df_contract_features = df_contract.merge(df_features, how='left', left_on='address', right_on='address')
df_contract_features = df_contract_features.drop(['IsContract_x','IsContract_y'],axis=1)
df_contract_features = df_contract_features.fillna(0)
df_contract_features = df_contract_features.drop(['address'], axis=1)

print(df_contract_features.shape)
X = df_contract_features.values
print(X[:2])
# 初始化K均值模型并拟合数据
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# 获取簇中心点和每个样本所属的簇标签
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Cluster Centers:\n", centroids)
print("\nCluster Labels:\n", labels)

# 添加新列到原始 DataFrame 中
df_contract = df_contract.assign(Label=labels)
df_contract = df_contract.drop(['IsContract'],axis=1)
print("Original Feature Matrix with Cluster Labels:")
print(df_contract)
df_contract.to_csv("../data/{}/contractAddress_add_cluter.csv".format(file_name),index=False)
