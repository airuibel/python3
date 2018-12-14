# -*- encoding:utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 500)
pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 500)
np.set_printoptions(suppress=True)

sample_data_0405 = pd.read_csv("sample_data_h5_4_5.csv", encoding="utf-8")

# 根据分析特征的数理统计得出，部分特征分布变化差异较大，或缺少率较大并无特殊业务含义，故删除此类特征
sample_data_0405 = sample_data_0405.drop(
    [
        "applicant_on_same_wifi",
        "applicant_on_same_wifi2",
        "account_on_same_device",
        "callog_is_authorize_result",
        "callog_one_month_busyhour_call_duration",
        "callog_one_month_busyhour_call_times",
        "callog_one_month_callin_duration",
        "callog_one_month_callin_times",
        "callog_one_month_callout_duration",
        "callog_one_month_callout_times",
        "callog_one_month_idlehour_call_duration",
        "callog_one_month_idlehour_call_times",
        "callog_one_month_ld_callout_duration",
        "callog_one_month_ld_callout_times",
        "callog_one_month_liainson_call_duration",
        "callog_one_month_liainson_call_ratio",
        "callog_one_month_liainson_call_times",
        "callog_one_month_local_callout_duration",
        "callog_one_month_local_callout_times",
        "callog_one_month_missed_call_times",
        "callog_one_month_non_liainson_call_duration",
        "callog_one_month_non_liainson_call_ratio",
        "callog_one_month_non_liainson_call_times",
        "callog_three_month_parents_call_times",
        "callog_time_from_earliest_call",
        "callog_time_from_liainson_last_call",
        "callog_total_call_times",
        "baidu_b_p_level1_black_cnt",
        "baidu_b_p_level2_black_cnt",
        "baidu_b_p_level3_black_cnt",
        "baidu_b_p_score",
        "baidu_l_p_identity_matched_12m",
        "baidu_l_p_identity_matched_15d",
        "baidu_l_p_identity_matched_30d",
        "baidu_l_p_identity_matched_3m",
        "baidu_l_p_identity_matched_6m",
        "baidu_l_p_identity_matched_7d",
        "baidu_l_p_identity_matched_over12m",
        "baidu_l_p_identity_query_org_cnt_12m",
        "baidu_l_p_identity_query_org_cnt_15d",
        "baidu_l_p_identity_query_org_cnt_30d",
        "baidu_l_p_identity_query_org_cnt_3m",
        "baidu_l_p_identity_query_org_cnt_6m",
        "baidu_l_p_identity_query_org_cnt_7d",
        "baidu_l_p_identity_query_org_cnt_over12m",
        "baidu_l_p_identity_query_times_12m",
        "baidu_l_p_identity_query_times_15d",
        "baidu_l_p_identity_query_times_30d",
        "baidu_l_p_identity_query_times_3m",
        "baidu_l_p_identity_query_times_6m",
        "baidu_l_p_identity_query_times_7d",
        "baidu_l_p_identity_query_times_over12m",
        "baidu_l_p_identity_score",
        "baidu_l_p_name_matched_12m",
        "baidu_l_p_name_matched_15d",
        "baidu_l_p_name_matched_30d",
        "baidu_l_p_name_matched_3m",
        "baidu_l_p_name_matched_6m",
        "baidu_l_p_name_matched_7d",
        "baidu_l_p_name_matched_over12m",
        "baidu_l_p_name_query_org_cnt_12m",
        "baidu_l_p_name_query_org_cnt_15d",
        "baidu_l_p_name_query_org_cnt_30d",
        "baidu_l_p_name_query_org_cnt_3m",
        "baidu_l_p_name_query_org_cnt_6m",
        "baidu_l_p_name_query_org_cnt_7d",
        "baidu_l_p_name_query_org_cnt_over12m",
        "baidu_l_p_name_query_times_12m",
        "baidu_l_p_name_query_times_15d",
        "baidu_l_p_name_query_times_30d",
        "baidu_l_p_name_query_times_3m",
        "baidu_l_p_name_query_times_6m",
        "baidu_l_p_name_query_times_7d",
        "baidu_l_p_name_query_times_over12m",
        "baidu_l_p_name_score",
        "baidu_l_p_phone_matched_12m",
        "baidu_l_p_phone_matched_15d",
        "baidu_l_p_phone_matched_30d",
        "baidu_l_p_phone_matched_3m",
        "baidu_l_p_phone_matched_6m",
        "baidu_l_p_phone_matched_7d",
        "baidu_l_p_phone_matched_over12m",
        "baidu_l_p_phone_query_org_cnt_12m",
        "baidu_l_p_phone_query_org_cnt_15d",
        "baidu_l_p_phone_query_org_cnt_30d",
        "baidu_l_p_phone_query_org_cnt_3m",
        "baidu_l_p_phone_query_org_cnt_6m",
        "baidu_l_p_phone_query_org_cnt_7d",
        "baidu_l_p_phone_query_org_cnt_over12m",
        "baidu_l_p_phone_query_times_12m",
        "baidu_l_p_phone_query_times_15d",
        "baidu_l_p_phone_query_times_30d",
        "baidu_l_p_phone_query_times_3m",
        "baidu_l_p_phone_query_times_6m",
        "baidu_l_p_phone_query_times_7d",
        "baidu_l_p_phone_query_times_over12m",
        "baidu_l_p_phone_score",
    ],
    axis=1,
)

sample_data_0405 = sample_data_0405.fillna(-1)

y = sample_data_0405["label"]
x = sample_data_0405.drop(["label", "loan_id"], axis=1)

# 切分训练集，测试集  7:3
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
    "booster": "gbtree",
    "objective": "binary:logistic",  # 多分类的问题    multi:softprob 返回值为概率  ，multi:softmax 返回值为整数 , reg:linear 线性回归问题
    # 'objective': 'binary:logistic',  # 二分类的逻辑回归问题，输出为概率
    # "num_class": 2,  # 类别数，与 multi:softmax 并用
    "eval_metric": "error",  # multi:softprob  ，multi:softmax  使用 merror
    "gamma": 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2
    "max_depth": 8,  # 构建树的深度，越大越容易过拟合
    "lambda": 12,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    "subsample": 1,  # 随机采样训练样本
    "colsample_bytree": 1,  # 生成树时进行的列采样
    "min_child_weight": 3,  # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative
    "silent": 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    "eta": 0.1,  # 如同学习率
    "seed": 0,
    # "scale_pos_weight": 2,  #  如果你关心的预测的ranking order（AUC)： – 通过scale_pos_weight来平衡正负类的权重 – 使用AUC进行评估,控制正负样本权重的平衡。一般对非平衡的很有用。一个很特殊的取值是：负样本个数/正样本个数。
    "max_delta_step": 1,  # 若关注预测出正确的概率值，这种情况下不能调整数据集的权重，可以通过设置参数max_delta_step为一个有限值比如1来加速模型训练的收敛,如果你关心的是预测的正确率： – 不能再平衡（re-balance）数据集 – 将参数max_delta_step设置到一个有限的数（比如：1）可以获得效果提升
    "nthread": 4,  # cpu 线程数
}

watchlist = [(dtrain, "train"), (dtest, "test")]
num_round = 150  # boosting迭代计算次数
plst = params.items()

model = xgb.train(plst, dtrain, num_round, evals=watchlist)
feature_impotance = model.get_fscore()
print(feature_impotance)
feature_impotance = sorted(feature_impotance.items(), key=lambda x: x[1], reverse=True)
feature_impotance_xgb = pd.DataFrame(feature_impotance)
print(feature_impotance_xgb)
# feature_impotance_xgb.to_csv("feature_impotance_xgb.csv",sep=",",index=False,header=True)


xx_test = xgb.DMatrix(X_test)
ans = model.predict(xx_test)
print(ans)
print(type(ans))


test_auc = metrics.roc_auc_score(y_test, ans)  # 测试集上的auc值
print("test数据集上的AUC值为 ： ", test_auc)


# 计算KS值
fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test), ans)
print("test数据集的XGB_KS : ", max(tpr - fpr))

plt.title("KS:{}".format(max(tpr - fpr)))
plt.plot(tpr)
plt.plot(fpr)
plt.show()

ans_s = (ans >= 0.5) * 1
print(ans_s)
cm = metrics.confusion_matrix(y_test, ans_s)  # 混淆矩阵 可计算精准度
print(cm)


# 准备报告
"""
将y_verify的index重新编排，为了和后面预测的指标合并
"""
y_verify = y_test.values
y_verify = pd.DataFrame(y_verify)

"""
将numpy.array 转为DataFrame
"""
ans2 = ans.tolist()
ans2 = pd.DataFrame(ans2)
df_data = pd.concat([y_verify, ans2], axis=1)
df_data.columns = ["y_verify", "ans2"]

print(df_data.head(1000))

cutoff = []
for i in np.arange(0, 1, 0.001):
    cutoff.append(round(i, 3))
cutoff = sorted(cutoff, reverse=True)
print(cutoff)


def to_report(df, cutoffs):
    """
    :param df: 带有真实标签和预测标签的数据框
    :param cutoffs: 切分点,顺序或倒序 排序后的
    :return:DataFrame
    """
    bad_true = df[df["y_verify"] == 1].shape[0]
    good_true = df[df["y_verify"] == 0].shape[0]
    data_all = df.shape[0]
    good_bad_list = []
    good_list = []
    bad_list = []
    precision_rate_list = []
    good_bad_rate_list = []
    total_bad_rejt_rate_list = []
    total_good_rejt_rate_list = []
    for x in cutoffs:
        good_bad = df[df["ans2"] >= x].shape[0]
        good_bad_list.append(good_bad)
        good_bad_rate = round((good_bad / data_all) * 1.0, 6)
        good_bad_rate_list.append(good_bad_rate)
        good = df[(df["ans2"] >= x) & (df["y_verify"] == 0)].shape[0]
        good_list.append(good)
        bad = df[(df["ans2"] >= x) & (df["y_verify"] == 1)].shape[0]
        bad_list.append(bad)
        if good_bad == 0:
            precision_rate_list.append(1)
        else:
            precision_rate = round((bad / good_bad) * 1.0, 6)
            precision_rate_list.append(precision_rate)
        total_bad_rejt_rate = round((bad / bad_true) * 1.0, 6)
        total_bad_rejt_rate_list.append(total_bad_rejt_rate)
        total_good_rejt_rate = round((good / good_true) * 1.0, 6)
        total_good_rejt_rate_list.append(total_good_rejt_rate)
    data_df = pd.DataFrame(
        {"0": cutoffs, "1": good_bad_list, "2": good_list, "3": bad_list, "4": precision_rate_list, "5": good_bad_rate_list, "6": total_bad_rejt_rate_list, "7": total_good_rejt_rate_list}
    )
    return data_df


data_df = to_report(df_data, cutoff)
print(data_df.head(1000))
# data_df.to_csv("data_df_report.csv", sep=",", index=False, header=True)
# model.save_model('xgbt02.model')
# data_df.to_excel("data_df_report.xlsx",index=False)




