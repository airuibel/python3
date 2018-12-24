#-*- encoding:utf-8 -*-
import pandas as pd
import numpy  as np

'''
分布
'''

# 控制台输出
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 5000)
pd.set_option("display.width", 1000) # 控制台输出不换行
np.set_printoptions(suppress=True)

sample_data = pd.read_csv("sample_data.csv",encoding="utf-8")
data_type_dict = pd.read_csv("data_type_dict.csv",encoding="utf-8")

print(data_type_dict.head(5))

data_category = data_type_dict["feature"][data_type_dict['type'] == "category"]      # 将类别变量筛选出来
data_category_list = []
for i in data_category:
    data_category_list.append(i.strip())
print(data_category_list)

data_numerical = data_type_dict["feature"][data_type_dict['type'] == "numerical"]    # 将连续变量筛选出来
data_numerical_list = []
for i in data_numerical:
    data_numerical_list.append(i.strip())
print(data_numerical_list)

data_numerical2 = sample_data[data_numerical_list].describe().T
print(data_numerical2)
data_numerical2.to_csv("feature_statistics.csv",mode='a',sep="|",index=True,header=True)

data_category2= sample_data[data_category_list].astype(object).describe().T
print(data_category2)
data_category2.to_csv("feature_statistics.csv",mode='a',sep="|",index=True,header=True)


# data1 = sample_data[["nfcs_marriage"]]
# print(data1.describe())
#
# sample_data[["nfcs_marriage"]] = sample_data[["nfcs_marriage"]].astype(object)
#
# data2 = sample_data[["nfcs_marriage"]]
# print(data2.describe())

# data_faul = sample_data.describe()
# data_faul2 = data_faul.T
#
# print(sample_data.count())
# print(data_faul)
# print(sample_data.head(5))
# data_faul.to_csv("feature_statistics.csv",sep="|",index=True,header=True)








