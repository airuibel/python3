import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 5000)
pd.set_option("display.width", 1000) # 控制台输出不换行
pd.set_option('display.max_columns', 50) # 最大列数目
np.set_printoptions(suppress=True)

data_df = pd.read_csv("/Users/kaka/PycharmProjects/python3-douc/test_data.csv",encoding="utf-8")

print(data_df.head(10))

df2 = data_df

# N_distinct = len(list(set(df2["age"])))
# print(list(set(df2["age"])))
# print(N_distinct)


# (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)

# total = df2.groupby(["age"])["label"].count()
# total = pd.DataFrame({'total': total})
#
# bad = df2.groupby(["age"])["label"].sum()
# bad = pd.DataFrame({'bad': bad})
#
# regroup = total.merge(bad, left_index=True, right_index=True, how='left')
# regroup.reset_index(inplace=True)
# print(regroup)
# regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)     # 每一个类别的 坏样本率
#
# dicts = dict(zip(regroup["age"],regroup['bad_rate']))
# N = sum(regroup['total'])
# B = sum(regroup['bad'])
# overallRate = B * 1.0 / N    # 总的坏样本率
# print(overallRate)
# binBadRate = dicts
#
# colLevels = sorted(list(set(df2['age'])))
# groupIntervals = [[i] for i in colLevels]
# print(groupIntervals)
#
# special_attribute = []
# split_intervals = 5 - len(special_attribute)

# while (len(groupIntervals) > split_intervals):
#     chisqList = []
#     for k in range(len(groupIntervals) - 1):
#         temp_group = groupIntervals[k] + groupIntervals[k + 1]
#         print(temp_group)
#         df2b = regroup.loc[regroup['age'].isin(temp_group)]
#         # print(df2b)
#         badRate = sum(df2b['bad']) * 1.0 / sum(df2b['total'])           # 计算相邻两个类别的 坏样本率
#         # print(badRate)
#         df2b['good'] = df2b.apply(lambda x: x['total'] - x['bad'], axis=1)
#         goodRate = sum(df2b['good']) * 1.0 / sum(df2b['total'])         # 计算相邻两个类别的 好样本率
#         # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
#         df2b['badExpected'] = df2b['total'].apply(lambda x: x * badRate)
#         df2b['goodExpected'] = df2b['total'].apply(lambda x: x * goodRate)
#         print(df2b)
#         badCombined = zip(df2b['badExpected'], df2b['bad'])
#         goodCombined = zip(df2b['goodExpected'], df2b['good'])
#         badChi = [(i[0] - i[1]) ** 2 / i[0] for i in badCombined]
#         goodChi = [(i[0] - i[1]) ** 2 / i[0] for i in goodCombined]
#         chi2 = sum(badChi) + sum(goodChi)
#         print(chi2)
#         chisq = chi2
#         chisqList.append(chisq)
#     print(chisqList)
#     best_comnbined = chisqList.index(min(chisqList))
#     print(best_comnbined)
#     groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined + 1]
#     groupIntervals.remove(groupIntervals[best_comnbined + 1])
# print(groupIntervals)
# groupIntervals = [sorted(i) for i in groupIntervals]
# cutOffPoints = [max(i) for i in groupIntervals[:-1]]
# print(cutOffPoints)



def AssignBin(x, cutOffPoints,special_attribute=[]):
    '''
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)

def BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[col],regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)

def Chi2(df, total_col, bad_col):
    '''
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :return: 卡方值
    '''
    df2 = df.copy()
    # 求出df中，总体的坏样本率和好样本率
    badRate = sum(df2[bad_col])*1.0/sum(df2[total_col])
    df2['good'] = df2.apply(lambda x: x[total_col] - x[bad_col], axis = 1)
    goodRate = sum(df2['good']) * 1.0 / sum(df2[total_col])
    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df[total_col].apply(lambda x: x*badRate)
    df2['goodExpected'] = df[total_col].apply(lambda x: x * goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0]-i[1])**2/i[0] for i in badCombined]
    goodChi = [(i[0] - i[1]) ** 2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2



cutOffPoints = [27, 31, 37, 38]

groupedvalues = df2['age'].apply(lambda x: AssignBin(x, cutOffPoints))     # 分箱
df2['temp_Bin'] = groupedvalues

print(df2)

(binBadRate,regroup2) = BinBadRate(df2, 'temp_Bin', 'label')

print(binBadRate)
print(regroup2)

[minBadRate, maxBadRate] = [min(binBadRate.values()),max(binBadRate.values())]
while minBadRate ==0 or maxBadRate == 1:
    # 找出全部为好／坏样本的箱
    indexForBad01 = regroup2[regroup2['bad_rate'].isin([0, 1])].temp_Bin.tolist()
    print(indexForBad01)
    bin = indexForBad01[0]
    if bin == max(regroup2.temp_Bin):
        cutOffPoints = cutOffPoints[:-1]
    elif bin == min(regroup2.temp_Bin):
        cutOffPoints = cutOffPoints[1:]
    else:
        # 和前一箱进行合并，并且计算卡方值
        currentIndex = list(regroup2.temp_Bin).index(bin)
        prevIndex = list(regroup2.temp_Bin)[currentIndex - 1]
        df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
        (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', 'label')
        # chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
        chisq1 = Chi2(df2b, 'total', 'bad')
        print(chisq1)
        # 和后一箱进行合并，并且计算卡方值
        laterIndex = list(regroup2.temp_Bin)[currentIndex + 1]
        df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
        (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', 'label')
        # chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
        chisq2 = Chi2(df2b, 'total', 'bad')
        print(chisq2)
        if chisq1 < chisq2:
            cutOffPoints.remove(cutOffPoints[currentIndex - 1])
        else:
            cutOffPoints.remove(cutOffPoints[currentIndex])
        groupedvalues = df2['age'].apply(lambda x: AssignBin(x, cutOffPoints))
        df2['temp_Bin'] = groupedvalues
        (binBadRate, regroup2) = BinBadRate(df2, 'temp_Bin', 'label')
        [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]

print(cutOffPoints)





