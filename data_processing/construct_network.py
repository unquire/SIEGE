import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# 创建一个图对象
G = nx.Graph()

df = pd.read_csv("../data/origin_split_1/transaction.csv")
edges = list(df.apply(lambda row: (row['from'], row['to']), axis=1))
# print(edges)
# 添加节点和边
G.add_edges_from(edges)
# 找到连通分量
connected_components = list(nx.connected_components(G))

# 过滤掉节点数小于10的连通分量
large_connected_components = [comp for comp in connected_components if len(comp) >= 10]
num_large_connected_components = len(large_connected_components)
# 创建新的图对象来表示过滤后的连通分量
filtered_graph = nx.Graph()

for comp in tqdm(large_connected_components, total=num_large_connected_components):
    subgraph = G.subgraph(comp)
    if not nx.is_empty(subgraph):  # 检查子图是否非空
        filtered_graph.add_edges_from(subgraph.edges())
filtered_nodes = [node if node != 'None' else None for node in filtered_graph.nodes() if node is not None and node != 'None']
# print(filtered_nodes[:5])
num_edges = filtered_graph.number_of_edges()
print("当前网络的边数为:", num_edges)
# 从filtered_graph中获取边的from和to
filtered_edges = filtered_graph.edges()
# 将边的信息转换为DataFrame
filtered_df = pd.DataFrame(filtered_edges, columns=['from', 'to'])

# 将原始CSV文件与filtered_df进行合并
merged_df = pd.merge(df, filtered_df, on=['from', 'to'], how='inner')
# print("merged_df:",merged_df.head())
print(merged_df.shape)
print("剩余地址占全部地址的:",merged_df.shape[0]/df.shape[0])
# 创建DataFrame
df_node = pd.DataFrame(filtered_nodes, columns=['address'])
print(df_node.head())
print(df_node.shape)
#
# # 绘制原始图
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# pos = nx.spring_layout(G)  # 定义节点布局
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_color='black')  # 绘制图形
# plt.title('Original Graph')
#
# # 绘制过滤后的图
# plt.subplot(1, 2, 2)
# pos_filtered = nx.spring_layout(filtered_graph)  # 定义节点布局
# nx.draw(filtered_graph, pos_filtered, with_labels=True, node_color='lightgreen', node_size=500, font_size=12, font_color='black')  # 绘制图形
# plt.title('Filtered Graph')
#
# # 保存图像
# # plt.savefig('graph_comparison.png')
#
# plt.show()