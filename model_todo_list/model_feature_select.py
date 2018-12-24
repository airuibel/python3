#-*- encoding:utf-8 -*-
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit



sample_data = pd.read_csv("sample_data.csv",encoding="utf-8")
second_feature = pd.read_csv("feature_initial_filter_second.csv",encoding="utf-8")

second_feature_list = second_feature["feature"].tolist()

sample_data = sample_data[second_feature_list]
sample_data = sample_data.fillna(-1)

y = sample_data["label"]
x = sample_data.drop("label",axis=1)

# 切分训练集，测试集  6:2:2
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=42)

X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test,y_test,test_size=0.5,random_state=42)

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)

columns=X_train.columns

'''
# xgboost demo

params = {
    "booster": "gbtree",
    "objective": "binary:logistic",  # 多分类的问题    multi:softprob 返回值为概率  ，multi:softmax 返回值为整数
    # 'objective': 'binary:logistic',  # 二分类的逻辑回归问题，输出为概率
    # "num_class": 2,  # 类别数，与 multi:softmax 并用
    'eval_metric':'auc',
    "gamma": 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2
    "max_depth": 6,  # 构建树的深度，越大越容易过拟合
    "lambda": 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    "subsample": 1,  # 随机采样训练样本
    "colsample_bytree": 1,  # 生成树时进行的列采样
    "min_child_weight": 3,  # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative
    "silent": 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    "eta": 0.08,  # 如同学习率
    "seed": 0,
    "nthread": 4,  # cpu 线程数
}

watchlist = [(dtrain,'train'),(dtest,'test')]
num_round = 150  # boosting迭代计算次数
plst = params.items()

model = xgb.train(plst,dtrain,num_round,evals=watchlist)

feature_impotance = model.get_fscore()
feature_impotance = sorted(feature_impotance.items(), key=lambda x:x[1],reverse=True)
print(feature_impotance)
feature_impotance_xgb = pd.DataFrame(feature_impotance)
# feature_impotance_xgb.to_csv("feature_impotance_xgb.csv",index=False,header=False)

xx_test = xgb.DMatrix(X_test1)
ans = model.predict(xx_test)

test_auc = metrics.roc_auc_score(y_test1, ans)  # 测试集上的auc值
print(test_auc)

# 计算KS值
fpr,tpr,thresholds=metrics.roc_curve(np.array(y_test1),ans)
print('XGB_KS:',max(tpr-fpr))

plt.title("KS:{}".format(max(tpr-fpr)))
plt.plot(tpr)
plt.plot(fpr)
plt.show()

ans = (ans >= 0.5)*1
print(ans)
cm = metrics.confusion_matrix(y_test1,ans)     # 混淆矩阵 可计算精准度
print(cm)
'''
###############################################################################
# 用另外的样本测试

# xx_test2 = xgb.DMatrix(X_test2)
# ans2 = model.predict(xx_test2)
#
# test_auc2 = metrics.roc_auc_score(y_test2, ans2)  # 测试集上的auc值
# print(test_auc2)
#
# # 计算KS值
# fpr,tpr,thresholds=metrics.roc_curve(np.array(y_test2),ans2)
# print('XGB_KS:',max(tpr-fpr))
#
# plt.title("KS:{}".format(max(tpr-fpr)))
# plt.plot(tpr)
# plt.plot(fpr)
# plt.show()
#
# ans2 = (ans2 >= 0.5)*1
# print(ans2)
# cm2 = metrics.confusion_matrix(y_test2,ans2)     # 混淆矩阵 可计算精准度
# print(cm2)

###########################################################################

'''
# 使用minepy做特征选择
from minepy import MINE
m = MINE(alpha=0.6, c=15, est="mic_approx")
m.compute_score(x['baidu_score'],y)

def print_stats(mine):
    print("MIC", mine.mic())             # 返回最大信息系数（MIC或MIC_e）
    print("MAS", mine.mas())             # 返回最大不对称分数（MAS）。
    print("MEV", mine.mev())             # 返回最大边值（MEV）
    print("MCN (eps=0)", mine.mcn(0))    # 返回eps> = 0的最小单元格数（MCN）
    print("MCN (eps=1-MIC)", mine.mcn_general())    # 返回eps = 1 - MIC的最小单元格编号（MCN）。
    print("GMIC", mine.gmic())                      # 返回广义最大信息系数（GMIC）。
    print("TIC", mine.tic())                        # 返回总信息系数（TIC或TIC_e）。如果norm == True TIC将在[0,1]中标准化。

print("Without noise:")
print_stats(m)
'''

#############################################################################

'''
# 随机森林

feat_labels=x.columns.tolist()
forest=RandomForestClassifier(n_estimators=300,
                          max_depth=6,
                          min_samples_split=10,
                          min_samples_leaf=10,
                          n_jobs=-1,random_state=0)
forest.fit(X_train,y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]

importances_dict = {}
for f in range(X_train.shape[1]):
    #评估特征重要性
    print(feat_labels[indices[f]],importances[indices[f]])
    importances_dict[feat_labels[indices[f]]] = importances[indices[f]]

x1 = pd.DataFrame([importances_dict]).T.reset_index()
x1.sort_values([0],ascending=False,inplace=True)    # ascending=False 倒序  inplace=True 改变原DataFrame
print(x1)
x1.to_csv("feature_impotance_rdmforest.csv",index=False,header=False)

pred = forest.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,pred))

# 计算KS值
fpr,tpr,thresholds=metrics.roc_curve(np.array(y_test),pred)
print('RandomForest_KS:',max(tpr-fpr))

plt.title("RandomForest_KS:{}".format(max(tpr-fpr)))
plt.plot(tpr)
plt.plot(fpr)
plt.show()
ans = (pred >= 0.5)*1
print(ans)
cm = metrics.confusion_matrix(y_test,ans)     # 矩阵
print(cm)

'''
