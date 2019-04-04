# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# 控制台输出不换行
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 5000)
pd.set_option("display.width", 10000)

df = pd.read_csv("data_model.csv", encoding="utf-8")

# 切分训练集，测试集
trainData, testData = train_test_split(df, test_size=0.3)

params = {
    "booster": "gbtree",
    "objective": "multi:softprob",  # 多分类的问题    multi:softprob 返回值为概率  ，multi:softmax 返回值为整数
    # 'objective': 'binary:logistic',  # 二分类的逻辑回归问题，输出为概率
    "num_class": 2,  # 类别数，与 multisoftmax 并用
    "gamma": 0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    "max_depth": 5,  # 构建树的深度，越大越容易过拟合
    "lambda": 1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    "subsample": 1,  # 随机采样训练样本
    "colsample_bytree": 1,  # 生成树时进行的列采样
    "min_child_weight": 3,  # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative
    "silent": 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    "eta": 0.1,  # 如同学习率
    "seed": 0,
    "nthread": 4,  # cpu 线程数
}

num_round = 200  # boosting迭代计算次数
plst = params.items()

x_train = trainData.drop(["is_over"], axis=1).as_matrix()
y_train = trainData.loc[:, "is_over"].as_matrix()
x_test = testData.drop(["is_over"], axis=1).as_matrix()
y_test = testData.loc[:, "is_over"].as_matrix()                # .as_matrix()  将数据框数据结构转换为使用数组的数据结构

dtrain = xgb.DMatrix(x_train, y_train)
# evallist = [(dtrain, 'train')]

bst = xgb.train(plst, dtrain, num_round)


# 对测试集进行预测
dtest = xgb.DMatrix(x_test)
ans = bst.predict(dtest)

# xx = pd.DataFrame([{"f0":552,"f1":528,"f2":120}]).as_matrix()
# print(xx)
# dtest = xgb.DMatrix(xx)
# yyy = bst.predict(dtest)
# print(yyy)


test_auc = metrics.roc_auc_score(y_test, ans[:,1])  # 验证集上的auc值
print(test_auc)


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
ceate_feature_map(trainData.drop(["is_over"], axis=1).columns)    #特征名列表

fig,ax = plt.subplots()
fig.set_size_inches(60,30)
xgb.plot_tree(bst,ax = ax,num_trees=100,fmap='xgb.fmap')
fig.savefig('xgb_tree.png')
xgb.plot_importance(bst)
plt.show()

# 计算KS值
fpr,tpr,thresholds=metrics.roc_curve(np.array(y_test),ans[:,1])
print('KS:',max(tpr-fpr))

plt.title("KS:{}".format(max(tpr-fpr)))
plt.plot(tpr)
plt.plot(fpr)
plt.show()



# 通过如下方式可以加载模型：
# bst = xgb.Booster({'nthread':4}) # init model
# bst.load_model("model.bin")      # load data

# 可以加载模型：
# tar = xgb.Booster(model_file='xgb.model')

