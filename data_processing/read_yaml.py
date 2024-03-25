import yaml
import pandas as pd

# 打开 YAML 文件并加载数据
with open('../data/data/urls.yaml', 'r') as file:
    data = yaml.safe_load(file)

# 过滤符合条件的字段
filtered_values = [item for item in data if item['category'] == 'Phishing']
print(filtered_values)
# 将过滤后的数据保存到 DataFrame 中
# df = pd.DataFrame(filtered_values, columns=['addresses'])
# pd.set_option('display.max_columns', None)
# df = df.dropna(subset=['addresses'])
# print(df)
# # 获取 'ETH' 键对应的地址列表
# # 提取地址并存储到列表中
# addresses = []
# for item in df['addresses']:
#     for address_list in item.values():
#         addresses.extend(address_list)
#
# # 创建包含地址的 DataFrame
# df_addresses = pd.DataFrame({'address': addresses, 'label': 1})
#
# print(df_addresses)
# df_addresses.to_csv("../data/PhishingAddress.csv",index=False)
# 将 DataFrame 保存到 CSV 文件中
# df.to_csv('filtered_data.csv', index=False)