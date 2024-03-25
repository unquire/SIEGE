import pandas as pd

# 读取两个 CSV 文件为 DataFrame
df1 = pd.read_csv('../data/PhishingAddress.csv')
df2 = pd.read_csv('../data/origin_split_2/address.csv')

# 使用 merge 方法将两个 DataFrame 合并
merged_df = pd.merge(df1, df2, how='inner')  # 使用内连接合并，只保留重叠部分
print("merged_df:",merged_df)
# 计算合并后的 DataFrame 的行数，即重合行的数量
overlap_count = len(merged_df)

# 打印重合行的数量
print("Number of overlapping rows:", overlap_count)
print("df1.shape:",df1.shape[0])
print("df2.shape:",df2.shape[0])
print("重合地址占比:",overlap_count/df1.shape[0])