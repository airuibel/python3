# -*- encoding:utf-8 -*-
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


scaler = preprocessing.StandardScaler()  # (原始数值−均值)/标准差
age_fare = df[["Age", "Fare"]].as_matrix()  # 返回二维array
scaler.fit(age_fare)
scaled_age_fare = scaler.transform(age_fare)
df["Age_scaled"] = scaled_age_fare[:, 0]
df["Fare_scaled"] = scaled_age_fare[:, 1]


train_columns = []
for x in df.columns:
    if x not in ["Age", "PassengerId", "Fare", "Name", "Ticket"]:
        train_columns.append(x)

train_df = df[train_columns]
train_np = train_df.as_matrix()

y = train_np[:, 0]
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(penalty='l1', tol=1e-6, C=1.0)
clf.fit(X,y)
print(clf)





# 测试样本

data_test = pd.read_csv("test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


age_fare2 = df_test[["Age", "Fare"]].as_matrix()  # 返回二维array
scaler.fit(age_fare2)
scaled_age_fare = scaler.transform(age_fare2)
df_test["Age_scaled"] = scaled_age_fare[:, 0]
df_test["Fare_scaled"] = scaled_age_fare[:, 1]


test_columns = []
for x in df.columns:
    if x not in ["Age", "PassengerId", "Fare", "Name", "Ticket","Survived"]:
        test_columns.append(x)

test = df_test[test_columns]
predictions = clf.predict(test)
predictions2 = pd.Series(predictions)
print(predictions.astype(np.int32))
result = pd.DataFrame({'PassengerId':data_test['PassengerId'],'rr':predictions2})
result.to_csv("logistic_regression_predictions.csv",index=False)








