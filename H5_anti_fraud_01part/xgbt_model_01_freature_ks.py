#-*- encoding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import metrics


pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 5000)
pd.set_option("display.width", 1000)  # 控制台输出不换行
np.set_printoptions(suppress=True)

feature_dict = pd.read_csv("feature_dict.csv",encoding="utf-8")
feature_dict = feature_dict.set_index("0").T.to_dict('list')   # 将数据框转化为字典
print(feature_dict)



feature_data = pd.read_csv("feature_impotance_xgb.csv",encoding="utf-8")
df_data = pd.read_csv("sample_data_h5_4_5.csv",encoding="utf-8")
df_data = df_data.fillna(-1)

feature_data = feature_data["0"].tolist()

def KS_cr(df,feature_datas,labels):
    """
    :param df: 完整样本的 DataFrame
    :param feature_datas: 需要计算KS的特征list
    :param labels: 标签名字
    :return: 计算好的特征KS DataFrame
    """
    good = df[df[labels] == 0].shape[0]
    bad = df[df[labels] == 1].shape[0]
    ks_list=[]
    feature_dict_list = []
    for x in feature_datas:
        print(x)
        if x in feature_dict.keys():
            pass
        else:
            feature_dict[x] = "暂时未添加数据字典"
        feature_dict_list.append(feature_dict[x])
        feature_ks = []
        df_s =df[[x,labels]]
        feature_vals = sorted(list(set(df_s[x].tolist())))     # 将特征的值 去重排序 顺序
        for i in feature_vals:
            a_bad = df_s[(df_s[x] <= i) & (df_s[labels] == 1)].shape[0]
            a_good = df_s[(df_s[x] <= i) & (df_s[labels] == 0)].shape[0]
            feature_ks.append(abs((a_bad/bad)-(a_good/good)))
        ks_list.append(max(feature_ks))
    df_ks = pd.DataFrame({"1":feature_datas,
                          "2":feature_dict_list,
                          "3":ks_list
    })
    return df_ks

df_ks = KS_cr(df_data,feature_data,"label")
print(df_ks)
# df_ks.to_csv("feature_ks.csv",sep=",",index=False,header=True)
































