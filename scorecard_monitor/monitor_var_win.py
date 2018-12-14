# -*- encoding:utf-8 -*-

import json
import pymongo
import pandas as pd
import configparser
import os
import datetime
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import DataFrame, Series

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # 现在
n_time = datetime.datetime.now().date() - datetime.timedelta(20)
# print(n_time)
time_array = time.strptime(str(n_time), "%Y-%m-%d")
time_stamp = int(time.mktime(time_array))
print(n_time)


# 建立MongoDB数据库连接
def connet_mongo(host_m, port_m, database_m, user_m, pasw_m, table_s):
    client = pymongo.MongoClient(host_m, port_m)
    db = client[database_m]
    db.authenticate(user_m, pasw_m)
    collection = db[table_s]
    return collection


class Server():
    def __init__(self, name, host, port, database, username, password):
        self.name = name
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password

    def __str__(self):
        return "%s , %s , %s , %s , %s , %s" % (
            self.name, self.host, self.port, self.database, self.username, self.password)


class Conf():
    def __init__(self, name, database, tables, columns, type, time_Identification, description, enable):
        self.name = name
        self.database = database
        self.tables = tables
        self.columns = columns
        self.type = type
        self.time_Identification = time_Identification
        self.description = description
        self.enable = enable

    def __str__(self):
        return "%s , %s , %s , %s , %s" % (
            self.database, self.tables, self.columns, self.type, self.time_Identification, self.description,
            self.enable)


os.chdir(os.getcwd())
config_server = configparser.ConfigParser()
config_server.read("server.ini")
sections = config_server.sections()

servers = dict()
for e in sections:
    servers[e] = Server(e,
                        config_server.get(e, 'host'),
                        config_server.getint(e, 'port'),
                        config_server.get(e, 'database'),
                        config_server.get(e, 'username'),
                        config_server.get(e, 'password')
                        )

conf = dict()
config_conf = configparser.ConfigParser()
config_conf.read("conf.ini", encoding="utf-8")
sections_s = config_conf.sections()

for e in sections_s:
    if config_conf.getint(e, 'enable') == 1:
        conf[e] = Conf(e,
                       config_conf.get(e, 'database'),
                       config_conf.get(e, 'tables'),
                       config_conf.get(e, 'columns'),
                       config_conf.getint(e, 'type'),
                       config_conf.get(e, 'time_Identification'),
                       config_conf.get(e, 'description'),
                       config_conf.getint(e, 'enable'),
                       )

# 生成mongo信息串,用于传递计算条件
transmit_compute = dict()
for jobs in conf.keys():
    job_list = conf[jobs]
    server_database = job_list.database
    # servers_job = servers[server_database]
    # print(job_list.database,job_list.tables,job_list.columns,job_list.type,job_list.time_Identification,job_list.description)
    try:
        if job_list.database not in transmit_compute:
            transmit_compute[job_list.database] = {}
            transmit_compute[job_list.database][job_list.tables] = [job_list.columns]
            transmit_compute[job_list.database][job_list.tables] = {}
            transmit_compute[job_list.database][job_list.tables][job_list.columns] = [job_list.description,
                                                                                      {'type': job_list.type}]
        else:
            if job_list.tables not in transmit_compute[job_list.database]:
                transmit_compute[job_list.database][job_list.tables] = {
                    job_list.columns: [job_list.description, {'type': job_list.type}]}
            else:
                transmit_compute[job_list.database][job_list.tables][job_list.columns] = [job_list.description,
                                                                                          {'type': job_list.type}]
    except BaseException as e:
        print("异常" + e)
# print(transmit_compute)

# 生成最终的mongo信息串,用于查询mongo数据
database_final = dict()
for jobs in conf.keys():
    job_list = conf[jobs]
    server_database = job_list.database
    # servers_job = servers[server_database]
    # print(job_list.database,job_list.tables,job_list.columns,job_list.type,job_list.time_Identification,job_list.description)
    try:
        if job_list.database not in database_final:
            database_final[job_list.database] = {}
            database_final[job_list.database][job_list.tables] = [job_list.columns]
            database_final[job_list.database][job_list.tables] = {}
            database_final[job_list.database][job_list.tables][job_list.columns] = [{'type': job_list.type},
                                                                                    job_list.description]
            database_final[job_list.database][job_list.tables]["time_Identification"] = [job_list.time_Identification]
        else:
            if job_list.tables not in database_final[job_list.database]:
                database_final[job_list.database][job_list.tables] = {
                    job_list.columns: [{'type': job_list.type}, job_list.description]}
                database_final[job_list.database][job_list.tables]["time_Identification"] = [
                    job_list.time_Identification]
            else:
                database_final[job_list.database][job_list.tables][job_list.columns] = [{'type': job_list.type},
                                                                                        job_list.description]
    except BaseException as e:
        print("异常" + e)

# 将mongo数据存在python内存中
datas = []

for a in database_final.keys():
    try:
        servers_job = servers[a]
        for b in database_final[a].keys():
            con_mongo = connet_mongo(servers_job.host, servers_job.port, servers_job.database, servers_job.username,
                                     servers_job.password, b)
            columns_list = {}
            time_s = {}
            for c in database_final[a][b].keys():
                if c != 'time_Identification':
                    columns_list[c] = 1
                else:
                    columns_list[database_final[a][b][c][0]] = 1
                    # time_key = list(json.loads(database_final[a][b][c][0]).keys())[0]
                    time_key = database_final[a][b][c][0]
                    time_s[time_key] = {"$gte": time_stamp}
            print("查询前---" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # 现在
            data = con_mongo.find(time_s, columns_list)
            # data = con_mongo.find(time_s)
            # print(con_mongo)
            # pd.set_option('display.max_rows', None)
            # pd.set_option('display.max_colwidth', 500)
            data_list = []
            for i in data:
                # 将所有时间的时间标识字段变为'logTime'
                i['logTime'] = i.pop(time_key)
                i['logTime'] = datetime.datetime.fromtimestamp(i['logTime']).date()
                i['logTime'] = time.strptime(str(i['logTime']), "%Y-%m-%d")
                i['logTime'] = int(time.mktime(i['logTime']))
                data_list.append(i)
            datas.append(data_list)
            print("查询后---" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # 现在
    except BaseException as e:
        print("配置文件出现异常：", e)


# print(len(datas))


# for x in range(len(datas)):
#     a = pd.DataFrame(datas[x])
#     # aa = a[a.age==22]
#     # aa = a.loc[a['age'] == 22]
#     print(a)


def getCurrentTime():
    str = time.strftime("%Y-%m-%d", time.localtime())
    date = time.strptime(str, "%Y-%m-%d")
    timeStamp = int(time.mktime(date))
    return timeStamp


databaseNameList = list(transmit_compute.keys())
# df = pd.DataFrame(datas)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
index = 0  # 将transmit_compute中的每个表和datas中的每个列表对应上
totalList = []
collectionFieldList = []  # 表和字段组成的list，用来生成html页面
for databaseNameList_index in range(len(databaseNameList)):
    databaseName = databaseNameList[databaseNameList_index]  # 'business'
    collectionDict = transmit_compute[databaseName]  # {'RulesEngineItem':{},'MergedData':{}}
    collectionNameList = list(collectionDict.keys())
    for collectionNameList_index in range(len(collectionNameList)):
        collectionName = collectionNameList[collectionNameList_index]  # 'RulesEngineItem'
        fieldDict = collectionDict[
            collectionName]  # {'amount': [{"极端":"0-3000,>=30001,null","中等":"3001-8000","大额":"8001-30000"},{'type':1}],'age': [{"未成年":"0-18,null","年轻人":"19-30","中年人":"31-45","老年人":">=46"},{'type':1}]}
        collectionRecordList = datas[index]
        # collectionRecordSeries = df.loc[index] # [{ 'amount': 30000, 'age': 38,'logTime':1524326400}, {'amount': 50000, 'age': 28,'logTime':1524326400}, {'amount': 10000, 'age': 22,'logTime':1524412800}],
        fieldNameList = list(fieldDict.keys())  # ['amount','age']
        # collectionRecordList = list(collectionRecordSeries)
        # print(collectionRecordList)
        # 找出logTime的最小值
        minLogTime = 4070880000  # 2099年1月1日的时间戳
        for i in range(len(collectionRecordList)):
            if collectionRecordList[i]['logTime'] < minLogTime:
                minLogTime = collectionRecordList[i]['logTime']
        # 将collectionSeries记录进行分组
        currentTimeStamp = getCurrentTime()  # 今天的时间戳
        tempTimeStamp = minLogTime
        perDayRecordList = []  # 每天数据的列表
        while tempTimeStamp <= currentTimeStamp:
            # 查询当天的记录保存到collectionRecordList中
            tempList = []
            for collectionRecordList_index in range(len(collectionRecordList)):
                if tempTimeStamp == collectionRecordList[collectionRecordList_index]['logTime']:
                    tempList.append(collectionRecordList[collectionRecordList_index])
            if len(tempList) != 0:
                perDayRecordList.append(tempList)
            tempTimeStamp = tempTimeStamp + 3600 * 24
        # 遍历计算每个属性的结果
        # print(perDayRecordList)
        for fieldNameList_index in range(len(fieldNameList)):
            try:
                fieldTotalList = []
                fieldName = fieldNameList[fieldNameList_index]  # 'amount'
                fieldType = fieldDict.get(fieldName)[1].get('type')  # 1
                fieldSectionDict = fieldDict.get(fieldName)[
                    0]  # {"极端":"0-3000,>=30001,null","中等":"3001-8000","大额":"8001-30000"}
                # print(type(fieldSectionDict))
                # print(fieldSectionDict)
                fieldSectionDict = json.loads(fieldSectionDict)
                fieldSectionList = list(fieldSectionDict.keys())
                # print(fieldSectionList)
                # 遍历每天的结果
                for perDayRecordList_index in range(len(perDayRecordList)):
                    oneDayRecordList = perDayRecordList[
                        perDayRecordList_index]  # [{'amount': 30000, 'age': 38, 'logTime': 1524326400},{'amount': 50000, 'age': 28, 'logTime': 1524326400}]
                    totalRecordCount = len(oneDayRecordList)
                    classTotalRecordCount = 0
                    sectionResultList = []  # 保存每天每个属性分段的记录
                    # 遍历计算每个属性分段的结果
                    for fieldSectionList_index in range(len(fieldSectionList)):
                        count = 0
                        statisticsFieldValueCond = fieldSectionDict[fieldSectionList[fieldSectionList_index]]
                        # 遍历某天所有记录数
                        for oneDayRecordList_index in range(len(oneDayRecordList)):
                            dataDict = oneDayRecordList[oneDayRecordList_index]
                            dataDictValue = dataDict.get(fieldName)  # 数据库amount字段的值，有可能是None
                            if fieldType == 1:
                                if ',' in statisticsFieldValueCond:
                                    statisticsFieldValueCondList = statisticsFieldValueCond.split(',')
                                    for statisticsFieldValueCondList_index in range(len(statisticsFieldValueCondList)):
                                        if (None != dataDictValue):
                                            if '-' in statisticsFieldValueCondList[statisticsFieldValueCondList_index]:
                                                statisticsFieldValueCondStr = statisticsFieldValueCondList[
                                                    statisticsFieldValueCondList_index]  # 0-3000
                                                if int(dataDictValue) >= int(
                                                        statisticsFieldValueCondStr[
                                                        0:statisticsFieldValueCondStr.index('-')]) and int(
                                                    dataDictValue) <= int(
                                                    statisticsFieldValueCondStr[
                                                    statisticsFieldValueCondStr.index('-') + 1:]):
                                                    # 满足条件
                                                    count = count + 1
                                            elif '>=' == statisticsFieldValueCondList[
                                                             statisticsFieldValueCondList_index][0:2]:
                                                if int(dataDictValue) >= int(
                                                        statisticsFieldValueCondList[
                                                            statisticsFieldValueCondList_index][2:]):
                                                    count = count + 1
                                            elif '<=' == statisticsFieldValueCondList[
                                                             statisticsFieldValueCondList_index][0:2]:
                                                if int(dataDictValue) <= int(
                                                        statisticsFieldValueCondList[
                                                            statisticsFieldValueCondList_index][2:]):
                                                    count = count + 1
                                            elif 'null' != statisticsFieldValueCondList[
                                                statisticsFieldValueCondList_index]:
                                                if int(dataDictValue) == int(
                                                        statisticsFieldValueCondList[
                                                            statisticsFieldValueCondList_index]):
                                                    count = count + 1
                                        else:
                                            if 'null' == statisticsFieldValueCondList[
                                                statisticsFieldValueCondList_index]:
                                                count = count + 1
                                else:
                                    if (None != dataDictValue):
                                        if '-' in statisticsFieldValueCond:
                                            if int(dataDictValue) >= int(
                                                    statisticsFieldValueCond[0:statisticsFieldValueCond.index(
                                                            '-')]) and int(dataDictValue) <= int(
                                                statisticsFieldValueCond[statisticsFieldValueCond.index('-') + 1:]):
                                                count = count + 1
                                        elif '>=' == statisticsFieldValueCond[0:2]:
                                            if int(dataDictValue) >= int(statisticsFieldValueCond[2:]):
                                                count = count + 1
                                        elif '<=' == statisticsFieldValueCond[0:2]:
                                            if int(dataDictValue) <= int(statisticsFieldValueCond[2:]):
                                                count = count + 1
                                        elif 'null' != statisticsFieldValueCond:
                                            if int(dataDictValue) == int(statisticsFieldValueCond):
                                                count = count + 1
                                    else:
                                        if 'null' == statisticsFieldValueCond:
                                            count = count + 1
                            elif fieldType == 0:
                                if (dataDictValue != None):
                                    if str(dataDictValue) in statisticsFieldValueCond:
                                        count = count + 1
                                else:
                                    if 'null' in statisticsFieldValueCond:
                                        count = count + 1
                        classTotalRecordCount = classTotalRecordCount + count
                        # 当天的dict
                        logTimeStamp = oneDayRecordList[0]['logTime']
                        logDate = time.strftime("%Y-%m-%d", time.localtime(logTimeStamp))
                        oneDayDict = {'logTime': logDate, 'collectionName': collectionName, 'class': fieldName,
                                      'classify': fieldSectionList[fieldSectionList_index], 'num': count}
                        fieldTotalList.append(oneDayDict)
                        sectionResultList.append(oneDayDict)
                    if classTotalRecordCount < totalRecordCount:
                        logTimeStamp = oneDayRecordList[0]['logTime']
                        logDate = time.strftime("%Y-%m-%d", time.localtime(logTimeStamp))
                        otherDict = {'logTime': logDate, 'collectionName': collectionName, 'class': fieldName,
                                     'classify': '其他', 'num': totalRecordCount - classTotalRecordCount}
                        fieldTotalList.append(otherDict)
                        sectionResultList.append(otherDict)
                    # 计算每天每个属性分段的概率
                    sectionCount = 0
                    for sectionResultList_index in range(len(sectionResultList)):
                        sectionCount = sectionCount + sectionResultList[sectionResultList_index]['num']
                    for sectionResultList_index in range(len(sectionResultList)):
                        sectionResultList[sectionResultList_index]['pct'] = str(
                            round(sectionResultList[sectionResultList_index]['num'] / sectionCount * 100, 2)) + '%'
                totalList.append(fieldTotalList)
                collectionFieldList.append(collectionName + "_" + fieldName)
            except BaseException as e:
                print('字段解析报错：')
                print(e)
        index = index + 1
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # 现在

from datetime import datetime,timedelta
# 定义作图基本参数
def barplot(df, save):
    # 设置子图框架
    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    plt.style.use('ggplot')
    df.plot(kind='bar', stacked=True, ax=ax, width=0.8656, legend=False)

    # 等分设置y轴刻度标签
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.2, 0.2), ['0%', '20%', '40%', '60%', '80%', '100%'], fontsize='medium')

    # 设置x轴时间间隔，按时间排序，以week为间隔
    dates_list = []
    for i in range(9):
        date = datetime.strptime(df.index[0],'%Y-%m-%d') + timedelta(days = -7 * i)
        date = date.strftime('%m-%d')
        dates_list.append(date)
    plt.xlim(-0.5, len(df) - 0.5)
    plt.xticks(np.arange(0, len(df) - 0.5, 7), dates_list, fontsize='medium')  # 时间间隔

    # 设置网格线条与图例，刻度标签位置
    ax.legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.,fontsize = 'xx-small')  # 设置ax中legend的位置,将其放在图外plt.show()
    ax.grid(axis='y', linestyle='-', linewidth=2, color='0.75', alpha=1.0)
    plt.xticks(rotation='horizontal')

    # 消除图像边缘和图像间的空白间隙，去除轴标题
    plt.tight_layout()
    ax.set_ylabel('')
    ax.set_xlabel('')

    # 设置输出路径
    #plt.savefig('/data/home/weplay/workspace/jupyter/monitor-var/data/home/weplay/workspace/jupyter/monitor-var' +save_name,ppi = '500', bbox_inches='tight')
    plt.savefig(save_name, ppi='500', bbox_inches='tight')

# 遍历列表，转换为DataFrame作图

for iter in totalList:
    save_name = iter[0].get('collectionName') + '_' + iter[0].get('class') + '.png'
    city = pd.DataFrame(iter).set_index(['logTime', 'classify'])
    city['pct'] = city['pct'].str.replace('%', '').astype(float) / 100
    # 转置后相当于按log_time,classify分类后取pct数值var
    city_stack = city.unstack()['pct'].sort_index(ascending=False)
    barplot(city_stack, save_name)


# 生成html页面
navbarNavStr = ''
dataSpyStr = ''
for collectionFieldList_index in range(len(collectionFieldList)):
    navbarNavStr = navbarNavStr + '<a class="' + collectionFieldList[collectionFieldList_index] + '" href="#' + \
                   collectionFieldList[collectionFieldList_index] + '"><span></span></a>'
    if collectionFieldList_index % 2 == 0:
        if collectionFieldList_index == len(collectionFieldList) - 1:
            dataSpyStr = dataSpyStr + '<div><div class="box-left"><div>' + '<span class="rule1" id="' + \
                         collectionFieldList[collectionFieldList_index] + '">' + collectionFieldList[
                             collectionFieldList_index] + '</span></div>' \
                                                          '<div style="width: 850px;height: 180px;><span class="rule2"><img src="' + \
                         collectionFieldList[collectionFieldList_index] + '.png"></span>' + '</div></div></div>'
        else:
            dataSpyStr = dataSpyStr + '<div><div class="box-left"><div>' + '<span class="rule1" id="' + \
                         collectionFieldList[collectionFieldList_index] + '">' + collectionFieldList[
                             collectionFieldList_index] + '</span></div>' \
                                                          '<div style="width: 850px;height: 180px;><span class="rule2"><img src="' + \
                         collectionFieldList[collectionFieldList_index] + '.png"></span>' + '</div></div></div>'
    else:
        dataSpyStr = dataSpyStr + '<div class="box-left"><div>' + '<span class="rule1" id="' + collectionFieldList[
            collectionFieldList_index] + '">' + collectionFieldList[collectionFieldList_index] + '</span></div>' \
                                                                                                 '<div style="width: 850px;height: 180px;><span class="rule2"><img src="' + \
                     collectionFieldList[collectionFieldList_index] + '.png"></span>' + '</div></div></div>'
htmlStr = '<!DOCTYPE html>' \
          '<html>' \
          '<head>' \
          '<meta charset="utf-8">' \
          '<title>风控监控工具</title>' \
          '<link rel="stylesheet" href="css/bootstrap.min.css">' \
          '<script src="js/jquery-3.3.1.min.js"></script>' \
          '<script src="js/bootstrap.min.js"></script>' \
          '<style type="text/css">' \
          '.box-left{ float:left; text-align:center;margin-left: 50px;} ' \
          '.box-right{ float:right; text-align:center;margin-right: 50px;} ' \
          '.search {margin-top:50px;margin-left:770px;margin-bottom:50px}' \
          '</style>' \
          '</head>' \
          '<body>' \
          '<script type="text/javascript">' \
          '$(function(){$("#btn").click(function(){content = $("#content").val();str = "." + content + ">span";$(str).trigger("click");})});</script>' \
          '<nav id="navbar-example">' \
          '<div class="search">' \
          '<input type="text" id="content" placeholder="请输入表名和特征" style="width: 300px"></input>' \
          '<button id="btn" type="button" style="margin-left: 5px;">search</button>' \
          '</div>' \
          '<div class="nav navbar-nav">' + navbarNavStr + '</div>' \
                                                          '</nav>' \
                                                          '<div data-spy="scroll" data-target="#navbar-example">' + dataSpyStr + '</div>' \
                                                                                                                                 '</body>' \
                                                                                                                                 '</html>'                                                                                                                                                                 ''

htmlf = open('monitor_vars.html', 'w+', encoding="utf-8")
htmlcont = htmlf.read()
htmlf.write(htmlStr)
htmlf.close
print("python程序执行完毕")













