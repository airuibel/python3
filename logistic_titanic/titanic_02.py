import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model


pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 500)
pd.set_option("display.width", 1000)  # 控制台输出不换行
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

data_train = pd.read_csv("train.csv")
data_train_info = data_train.info()
data_train_describe = data_train.describe()
print(data_train_info)
print(data_train_describe)


def set_missing_ages(df):
    age_df = df[["Age", "Pclass", "SibSp", "Parch", "Fare"]]
    known_age = age_df[age_df["Age"].notnull()].as_matrix()  # 此处的 as_matrix 函数是将数据框数据结构转换为使用数组的数据结构
    unknow_age = age_df[age_df["Age"].isnull()].as_matrix()

    # fit到RandomForestRegressor之中
    y = known_age[:, 0]
    x = known_age[:, 1:]
    rfr = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=0)
    rfr.fit(x, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknow_age[:, 1::])
    # a = df.loc[(df.Age.isnull()), 'Age'] 相当于 a = df['Age'][df['Age'].isnull()]
    # 用得到的预测结果填补原缺失数据
    df.loc[(df["Age"].isnull()), "Age"] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df["Cabin"].notnull()), "Cabin"] = "yes"
    df.loc[(df["Cabin"].isnull()), "Cabin"] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)


# 通过get_dummies进行one-hot编码
dummies_Cabin = pd.get_dummies(data_train["Cabin"], prefix="Cabin")
dummies_Embarked = pd.get_dummies(data_train["Embarked"], prefix="Embarked")
dummies_Sex = pd.get_dummies(data_train["Sex"], prefix="Sex")
dummies_Pclass = pd.get_dummies(data_train["Pclass"], prefix="Pclass")

# 合并数据
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)  # axis=1 表示合并列
# 删除多余的数据 (列)
df.drop(["Cabin", "Embarked", "Sex", "Pclass"], axis=1, inplace=True)  # axis=1 表示删除列

# pearson相关系数的检验
# d2=df.corr()["Survived"]
# print(d2)


# 卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


data_columns = []
for x in df.columns:
    if x not in ["Survived","Age","Fare", "Name", "Ticket"]:
        data_columns.append(x)

print(df.ix[0:1,data_columns])


model1 = SelectKBest(chi2, k=5)  #选择k个最佳特征
ss = model1.fit_transform(df[data_columns], df["Survived"])  #该函数可以选择出k个特征
s = model1.scores_     # 得分
s2 = model1.pvalues_   # P值
np.set_printoptions(suppress=True)   # 取消科学计数法
print(s)

test = pd.DataFrame([s2],columns=data_columns)
with pd.option_context('display.float_format', lambda x: '%.8f' % x):
    print(test)















