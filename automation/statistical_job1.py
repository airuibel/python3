import pandas as pd
import numpy as np
import pymongo

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 5000)
pd.set_option("display.width", 1000)  # 控制台输出不换行
np.set_printoptions(suppress=True)


def connet_mongo(host_m, port_m, database_m, user_m, pasw_m, table_s):
    client = pymongo.MongoClient(host_m, port_m)
    db = client[database_m]
    db.authenticate(user_m, pasw_m)
    collection = db[table_s]
    return collection


con = connet_mongo("172.20.211.148", 37016, "thirdparty", "wp_reader", "6HhNBT3zBerekctq", "BaiRongData")

template_date = pd.read_csv("xxx.csv", encoding="utf-8")
# print(template_date)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def all_null(s):
    try:
        data_fauls = pd.DataFrame(s)
        data_describes = data_fauls.describe()
        return True
    except ValueError:
        return False


def dicts(dict_s):
    if isinstance(dict_s, dict):
        for x in dict_s.keys():
            if x != "_id":
                dicts(dict_s[x])
    elif type(dict_s) == list:
        for y in dict_s:
            dicts(y)
    else:
        if isinstance(dict_s, str):
            if is_number(dict_s):
                if dict_s.find(".") == -1:
                    data_da2.append(int(dict_s))
                else:
                    a1 = dict_s.split(".")
                    aa = int(a1[0]) + int(a1[1]) * (0.1 ** len(a1[1]))
                    data_da2.append(aa)
            else:
                if dict_s.strip() == "":
                    pass
                else:
                    data_da2.append(dict_s)
        else:
            if np.isnan(dict_s):
                pass
            else:
                data_da2.append(dict_s)


data_name = template_date["name"]

for i in data_name:
    print(i)
    t = template_date["type"][template_date["name"] == i].values[0]
    d = template_date["desc"][template_date["name"] == i].values[0]
    time_s = {"logTime": {"$gte": 1519833600}}
    column_s = {}
    column_s[i] = 1
    mongo_data = con.find(time_s, column_s)
    data_num = []
    try:
        for xx in mongo_data:
            data_da2 = []
            data_da3 = []
            dicts(xx)
            data_num.append(data_da2)
        data_faul = pd.DataFrame(data_num)
        data_describe = data_faul.describe()
        if len(data_describe.index) > 4:
            str_int = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                i,
                t,
                d,
                data_describe[0]["count"],
                data_describe[0]["mean"],
                data_describe[0]["std"],
                data_describe[0]["min"],
                data_describe[0]["25%"],
                data_describe[0]["50%"],
                data_describe[0]["75%"],
                data_describe[0]["max"],
            )
            print(str_int)
            test = open("xxx_int.csv", "a+", encoding="utf-8")
            test.write(str_int)
            test.close()
        else:
            str_str = "{},{},{},{},{},{},{}\n".format(i, t, d, data_describe[0]["top"], data_describe[0]["freq"], data_describe[0]["unique"], data_describe[0]["count"])
            print(str_str)
            test = open("xxx_str.csv", "a+", encoding="utf-8")
            test.write(str_str)
            test.close()
    except BaseException as e:
        print(e)
        str_str = "{},{},{},{},{},{},{}\n".format(i, t, d, "未查得", "未查得", "未查得", "未查得")
        test = open("xxx_str.csv", "a+", encoding="utf-8")
        test.write(str_str)
        test.close()
