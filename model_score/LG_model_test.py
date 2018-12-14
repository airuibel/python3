#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
import math
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt


# 控制台输出不换行
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 5000)
pd.set_option("display.width", 10000)

"""
icekredit_huiyan_score is in processing
[480.0, 522.0, 558.0, 590.0, 603.0]
baidu_score is in processing
[502.0, 542.0, 574.0, 599.0, 618.0]
fm_final_score is in processing
[35.0, 62.0, 80.0]
"""

df = pd.read_csv("test_data.csv",encoding="utf-8")

print(df.head(10))

df.loc[df['icekredit_huiyan_score'] <= 480 , 'icekredit_huiyan_score_more'] = 0.30498945091854057
df.loc[(df['icekredit_huiyan_score'] > 480) & (df['icekredit_huiyan_score'] <= 522), 'icekredit_huiyan_score_more'] = 0.206464728131119
df.loc[(df['icekredit_huiyan_score'] > 522) & (df['icekredit_huiyan_score'] <= 558), 'icekredit_huiyan_score_more'] = 0.07333411639548436
df.loc[(df['icekredit_huiyan_score'] > 558) & (df['icekredit_huiyan_score'] <= 590), 'icekredit_huiyan_score_more'] = -0.0024278566893290855
df.loc[(df['icekredit_huiyan_score'] > 590) & (df['icekredit_huiyan_score'] <= 603), 'icekredit_huiyan_score_more'] = -0.31662848256589404
df.loc[(df['icekredit_huiyan_score'] > 603), 'icekredit_huiyan_score_more'] = -0.538835434850186

df.loc[df['baidu_score'] <= 502 , 'baidu_score_more'] = 0.938461325037141
df.loc[(df['baidu_score'] > 502) & (df['baidu_score'] <= 542), 'baidu_score_more'] = 0.19893900215936247
df.loc[(df['baidu_score'] > 542) & (df['baidu_score'] <= 574), 'baidu_score_more'] = 0.06840662170368793
df.loc[(df['baidu_score'] > 574) & (df['baidu_score'] <= 599), 'baidu_score_more'] = -0.21991519444411467
df.loc[(df['baidu_score'] > 599) & (df['baidu_score'] <= 618), 'baidu_score_more'] = -0.4087495245534352
df.loc[(df['baidu_score'] > 618), 'baidu_score_more'] = -0.4141628855419666


df.loc[df['fm_final_score'] <= 35 , 'fm_final_score_more'] = -0.12395148067585944
df.loc[(df['fm_final_score'] > 35) & (df['fm_final_score'] <= 62), 'fm_final_score_more'] = -0.03209536440716791
df.loc[(df['fm_final_score'] > 62) & (df['fm_final_score'] <= 80), 'fm_final_score_more'] = 0.19872509532090318
df.loc[(df['fm_final_score'] > 80), 'fm_final_score_more'] = 0.38234347451586675

print(df.head(10))


x_test = df[['icekredit_huiyan_score_more','baidu_score_more','fm_final_score_more']]
y_test = df.loc[:,'is_over']

cls = joblib.load('model_LG_test.pkl')

predict_proba_all = cls.predict_proba(x_test)  # 概率
predict_proba_over = predict_proba_all[:,1]     # 逾期概率
print(predict_proba_over)

# train_prob_y = cls.predict(x_test)
# train_auc = metrics.roc_auc_score(y_test,predict_proba_over)       #训练集上的auc值
# print(train_auc)
#
#
# # 计算KS值
# fpr,tpr,thresholds=metrics.roc_curve(np.array(y_test),predict_proba_over)
# print('KS:',max(tpr-fpr))
#
# plt.title("KS:{}".format(max(tpr-fpr)))
# plt.plot(tpr)
# plt.plot(fpr)
# plt.show()
#
# df["prd"] = train_prob_y









