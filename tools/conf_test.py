#-*- encoding:utf-8 -*-
import pymongo
import pandas  as  pd
import numpy   as  np
import datetime


pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 500)

# ,{"platform":1,"product":1,"gender":1,"resident_address_city":1}

client = pymongo.MongoClient('172.20.211.150',37014)
db = client['business']
db.authenticate("wp_reader", "6HhNBT3zBerekctq")
collection = db['RulesEngineItem']
print(collection)


# def connet_mongo(host_m, port_m, database_m, user_m, pasw_m, table_s):
#     client = pymongo.MongoClient(host_m, port_m)
#     db = client[database_m]
#     db.authenticate(user_m, pasw_m)
#     collection = db[table_s]
#     return collection


print('查询开始',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
data_mongo = collection.find({'logTime':{"$gte":1525017600}}).limit(10)
x =0
data_list = []
for i in data_mongo:
    data_list.append(i)
    x = x+1
    print(x)
df = pd.DataFrame(data_list)
print('查询结束',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

df_dec = df.describe()


conf_int = '''
[monitor-%s]\n
database = business\n
tables = RulesEngineItem\n
columns = %s\n
type = 1\n
time_identification = logTime\n
description = {"第一":"[%s,%s]","第二":"[%s,%s]","第三":"[%s,%s]","第四":">=%s"}\n
enable = 1\n
'''

conf_str = '''
[monitor-%s]\n
database = business\n
tables = RulesEngineItem\n
columns = %s\n
type = 0\n
time_identification = logTime\n
description = {"第一":"%s","第二":"%s","第三":"%s"}\n
enable = 1\n
'''

df_alls = [columns for columns in df.columns]
df_decs = [columns for columns in df_dec.columns]


for i in df_alls:
    ser_data = df[i]
    ser_list = list(ser_data)
    ser_l = list(set(ser_list))
    if len(ser_l) >= 10 and i in df_decs:
        # print(ser_l)
        conf_i = conf_int %(i,i,df_dec[i]['min'],df_dec[i]['25%'],df_dec[i]['25%'],df_dec[i]['50%'],df_dec[i]['50%'],df_dec[i]['75%'],df_dec[i]['75%'])
        test = open('conf_int.ini','a+',encoding='utf-8')
        test.write(conf_i)
        test.close()
    else:
        ds = df.groupby(i).size()
        ds_d = pd.DataFrame(ds)
        data_cnt2 = ds_d.sort_values([0], ascending=[False])
        data_cnt2_list = list(data_cnt2.index)
        for x in range(len(data_cnt2_list)):
            if data_cnt2_list[x] == '':
                data_cnt2_list[x] = 'null'
        if len(data_cnt2_list) == 1:
            conf_s = conf_str %(i,i,data_cnt2_list[0],'未知','未知')
        if len(data_cnt2_list) == 2:
            conf_s = conf_str %(i,i,data_cnt2_list[0],data_cnt2_list[1],'未知')
        if len(data_cnt2_list) >= 3:
            conf_s = conf_str % (i, i, data_cnt2_list[0],data_cnt2_list[1],data_cnt2_list[2])
            test2 = open('conf_str.ini','a+',encoding='utf-8')
            test2.write(conf_s)
            test2.close()


















