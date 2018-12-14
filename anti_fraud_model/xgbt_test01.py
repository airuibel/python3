#-*- encoding:utf-8 -*-
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 5000)
pd.set_option("display.width", 1000)  # 控制台输出不换行
np.set_printoptions(suppress=True)

# 载入数据样本
sample_data = pd.read_csv("sample_data02.csv",encoding="utf-8")

# 将缺失值用 -1 填补
sample_data = sample_data.fillna(-1)
# print(sample_data.head(5))

'''
由于有些特征的值都为NaN值，故需要删除 
'''
lines = sample_data.shape[0]
feature = sample_data.columns.tolist()
# print(feature)

for i in feature:
    print(i)
    ratio = (sample_data[i][sample_data[i] == -1].shape[0]/lines) * 1.0
    if ratio > 0.99:
        print("删除该特征")
        sample_data.drop([i],axis=1,inplace=True)

# print(sample_data.head(5))

y = sample_data["label"]
x = sample_data.drop(["label","application_id"],axis=1)

# 切分训练集，测试集  6:2:2
X_train, X_train2, y_train, y_train2 = train_test_split(x,y,test_size=0.4,random_state=0)

X_test, X_verify, y_test, y_verify = train_test_split(X_train2,y_train2,test_size=0.5,random_state=0)

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
dverify = xgb.DMatrix(X_verify,label=y_verify)

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
    "eta": 0.07,  # 如同学习率
    "seed": 0,
    "nthread": 4,  # cpu 线程数
}

watchlist = [(dtrain,'train'),(dtest,'test')]
num_round = 200  # boosting迭代计算次数
plst = params.items()

model = xgb.train(plst,dtrain,num_round,evals=watchlist)
feature_impotance = model.get_fscore()
feature_impotance = sorted(feature_impotance.items(), key=lambda x:x[1],reverse=True)
feature_impotance_xgb = pd.DataFrame(feature_impotance)
print(feature_impotance_xgb)

xx_test = xgb.DMatrix(X_test)
ans = model.predict(xx_test)

test_auc = metrics.roc_auc_score(y_test, ans)  # 测试集上的auc值
print("test数据集上的AUC值为 ： ",test_auc)


# 计算KS值
fpr,tpr,thresholds=metrics.roc_curve(np.array(y_test),ans)
print('test数据集的XGB_KS : ',max(tpr-fpr))

# plt.title("KS:{}".format(max(tpr-fpr)))
# plt.plot(tpr)
# plt.plot(fpr)
# plt.show()

ans_s = (ans >= 0.5)*1
print(ans_s)
cm = metrics.confusion_matrix(y_test,ans_s)     # 混淆矩阵 可计算精准度
print(cm)


# 用另外的样本测试

xx_verify = xgb.DMatrix(X_verify)
ans2 = model.predict(xx_verify)

test_auc2 = metrics.roc_auc_score(y_verify, ans2)  # 测试集上的auc值
print(test_auc2)

# 计算KS值
fpr,tpr,thresholds=metrics.roc_curve(np.array(y_verify),ans2)
print('XGB_KS:',max(tpr-fpr))

# plt.title("KS:{}".format(max(tpr-fpr)))
# plt.plot(tpr)
# plt.plot(fpr)
# plt.show()

ans2_s = (ans2 >= 0.5)*1
print(ans2_s)
cm2 = metrics.confusion_matrix(y_verify,ans2_s)     # 混淆矩阵 可计算精准度
print(cm2)

# 准备报告
'''
将y_verify的index重新编排，为了和后面预测的指标合并
'''
y_verify = y_verify.values
y_verify = pd.DataFrame(y_verify)

'''
将numpy.array 转为DataFrame
'''
ans2 = ans2.tolist()
ans2 = pd.DataFrame(ans2)
df_data = pd.concat([y_verify,ans2],axis=1)
df_data.columns = ["y_verify","ans2"]
print(df_data.head(10))

cutoff = []
for i in np.arange(0,1,0.001):
    cutoff.append(i)
cutoff = sorted(cutoff,reverse=True)
print(cutoff)

def to_report(df,cutoffs):
    '''
    :param df: 带有真实标签和预测标签的数据框
    :param cutoffs: 切分点,顺序或倒序 排序后的
    :return:DataFrame
    '''
    bad_true = df[df["y_verify"] == 1].shape[0]
    good_true = df[df["y_verify"] == 0].shape[0]
    good_bad_list = []
    good_list = []
    bad_list = []
    precision_rate_list = []
    total_bad_rejt_rate_list = []
    total_good_rejt_rate_list = []
    for x in cutoffs:
        good_bad = df[df["ans2"] >= x].shape[0]
        good_bad_list.append(good_bad)
        good = df[(df["ans2"] >= x) & (df["y_verify"] ==0)].shape[0]
        good_list.append(good)
        bad = df[(df["ans2"] >= x) & (df["y_verify"] ==1)].shape[0]
        bad_list.append(bad)
        if good_bad == 0:
            precision_rate_list.append(1)
        else:
            precision_rate = (bad/good_bad) * 1.0
            precision_rate_list.append(precision_rate)
        total_bad_rejt_rate = (bad/bad_true) * 1.0
        total_bad_rejt_rate_list.append(total_bad_rejt_rate)
        total_good_rejt_rate = (good/good_true) * 1.0
        total_good_rejt_rate_list.append(total_good_rejt_rate)
    data_df = pd.DataFrame({"0":cutoffs,
                            "1":good_bad_list,
                            "2":good_list,
                            "3":bad_list,
                            "4":precision_rate_list,
                            "5":total_bad_rejt_rate_list,
                            "6":total_good_rejt_rate_list})
    return data_df

data_df = to_report(df_data,cutoff)
print(data_df.head(1000))

