#-*- encoding:utf-8 -*-

import datetime
import pandas  as  pd
import pymongo
import time
import numpy as np
import configparser
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

print("监控开始",datetime.datetime.now())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 500)
np.set_printoptions(suppress=True)


# 定义比较级
class Functions():
    ALIAS = {
        '=': 'eq',
        '!=': 'neq',
        '>': 'gt',
        '>=': 'gte',
        '<': 'lt',
        '<=': 'lte',
    }

def eq(*args):
    return [args[0] == args[1],"等于"]


def neq(*args):
    return [args[0] != args[1],"不等于"]


def gt(*args):
    return [args[0] > args[1],"大于"]


def gte(*args):
    return [args[0] >= args[1],"超过"]


def lt(*args):
    return [args[0] < args[1],"小于"]


def lte(*args):
    return [args[0] <= args[1],"低于"]

# 邮件发送

my_sender = 'welabreport@wolaidai.com'
my_pass = 'Qv@xYZ9Ed%Y*JxH8'
# my_user = ['kei.wong@wolaidai.com','sherry.jiang@wolaidai.com','amy.chen@wolaidai.com','kristal.song@wolaidai.com','rochelle.huang@wolaidai.com','kaka.yao@wolaidai.com','arvin.huo@wolaidai.com','william.wang@wolaidai.com','chace.ye@wolaidai.com']
my_user = ['kaka.yao@wolaidai.com']
my_admin = ['kaka.yao@wolaidai.com','arvin.huo@wolaidai.com','william.wang@wolaidai.com','chace.ye@wolaidai.com']
my_user = ','.join(my_user)
my_admin = ','.join(my_admin)

def e_mail(to_email,em_text):
    ret = True
    try:
        msg = MIMEText(em_text, 'plain', 'utf-8')
        msg['From'] = formataddr(["WeLab Report<welabreport@wolaidai.com>", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        # msg['To'] = formataddr(["测试收件人", to_email])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['to'] = to_email
        msg['Subject'] = "监控预警"  # 邮件的主题，也可以说是标题

        server = smtplib.SMTP_SSL("smtp.exmail.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender, to_email.split(','), msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()
    except Exception:
        ret = False
    return ret

mail_str = """告警!!!\n"""
mail_add = """         过去{}小时，{}，{} 中为 {} 的值占总量的 ({})，{} 阈值 ({})。\n"""
mail_admin = """监控发生异常，{} 部分发生异常，报错信息为({}) ,详细信息请查阅代码日志"""



# 如若有新的评分卡上线或者对应的score_card_version变化时，需要修改此字典
platform_scv = {"3.2":"IOS评分卡",
                "3.3":"Android评分卡",
                "6.0":"H5移动评分卡",
                "4.1":"H5联通评分卡",
                "7.0":"H5电信评分卡"
                }


# 建立MongoDB数据库连接
def connet_mongo(host_m, port_m, database_m, user_m, pasw_m, table_s):
    client = pymongo.MongoClient(host_m, port_m)
    db = client[database_m]
    db.authenticate(user_m, pasw_m)
    collection = db[table_s]
    return collection


# mongo 数据库信息
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
    def __init__(self, name, database, table, score_card_version, time_period,column, values, alarm,compare,enable):
        self.name = name
        self.database = database
        self.table = table
        self.score_card_version = score_card_version
        self.time_period = time_period
        self.column = column
        self.values = values
        self.alarm = alarm
        self.compare = compare
        self.enable = enable

    def __str__(self):
        return "%s , %s , %s , %s , %s, %s ,%s,%s,%s" % (
            self.database, self.table, self.score_card_version, self.time_period,self.column, self.values,self.alarm,self.compare,self.enable)

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
config_conf.read("monitoring_conf.ini", encoding="utf-8")
sections_s = config_conf.sections()


for e in sections_s:
    if config_conf.getint(e, 'enable') == 1:
        try:
            conf[e] = Conf(e,
                           config_conf.get(e, 'database'),
                           config_conf.get(e, 'table'),
                           config_conf.get(e, 'score_card_version'),
                           config_conf.getint(e, 'time_period'),
                           config_conf.get(e, 'column'),
                           config_conf.get(e, 'values'),
                           config_conf.get(e, 'alarm'),
                           config_conf.get(e, 'compare'),
                           config_conf.getint(e, 'enable'),
                           )
        except BaseException as e:
            admin_str1 = mail_admin.format("配置文件",e)
            e_mail(my_admin,admin_str1)
    else:
        pass

#解析配置文件，得出所需字典
monitoring_detail = {}
for jobs in conf.keys():
    try:
        job_s = conf[jobs]
        if job_s.database not in monitoring_detail.keys():
            monitoring_detail[job_s.database] = {}
            monitoring_detail[job_s.database][job_s.table] = {}
            monitoring_detail[job_s.database][job_s.table][job_s.score_card_version] = {}
            monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period] = {}
            monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period][job_s.column] ={}
            monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period][job_s.column][job_s.values] = {}
            monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period][job_s.column][job_s.values][job_s.compare] = job_s.alarm
        else:
            if job_s.table not in monitoring_detail[job_s.database].keys():
                monitoring_detail[job_s.database][job_s.table]={job_s.score_card_version:{job_s.time_period:{job_s.column:{job_s.values:{job_s.compare:job_s.alarm}}}}}
            else:
                if job_s.score_card_version not in monitoring_detail[job_s.database][job_s.table].keys():
                    monitoring_detail[job_s.database][job_s.table][job_s.score_card_version]={job_s.time_period:{job_s.column:{job_s.values:{job_s.compare:job_s.alarm}}}}
                else:
                    if job_s.time_period not in monitoring_detail[job_s.database][job_s.table][job_s.score_card_version].keys():
                        monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period] = {job_s.column:{job_s.values:{job_s.compare:job_s.alarm}}}
                    else:
                        if job_s.column not in monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period].keys():
                            monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period][job_s.column] = {job_s.values:{job_s.compare:job_s.alarm}}
                        else:
                            if job_s.values not in monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period][job_s.column].keys():
                                monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period][job_s.column][job_s.values] = {job_s.compare:job_s.alarm}
                            else:
                                monitoring_detail[job_s.database][job_s.table][job_s.score_card_version][job_s.time_period][job_s.column][job_s.values][job_s.compare] = job_s.alarm
    except BaseException as e:
        admin_str2 = mail_admin.format("解析配置文件",e)
        e_mail(my_admin,admin_str2)
        
        
# print(monitoring_detail)

email_open = 0

# 计算主体
for db in monitoring_detail.keys():
    try:
        servers_job = servers[db]
        for tb in monitoring_detail[db].keys():
            con_mongo = connet_mongo(servers_job.host, servers_job.port, servers_job.database, servers_job.username,
                                     servers_job.password, tb)
            mongo_cond = {}
            for scv in monitoring_detail[db][tb].keys():
                mongo_cond["score_card_version"] = scv
                # 将mongo数据撸到pandas中
                for time_p in monitoring_detail[db][tb][scv].keys():
                    dt = datetime.datetime.now() - datetime.timedelta(hours=time_p)
                    dt_int = int(time.mktime(time.strptime(dt.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')))
                    mongo_cond["logTime"] = {"$gte": dt_int}
                    mongo_comm = {}
                    for column_mg in monitoring_detail[db][tb][scv][time_p].keys():
                        mongo_comm[column_mg] = 1
                    mongo_data = con_mongo.find(mongo_cond,mongo_comm)
                    mongo_datas = []
                    for x in mongo_data:
                        mongo_datas.append(x)
                    data_df = pd.DataFrame(mongo_datas)
                    # print("###########",scv,time_p)
                    # 计算 df[df['total_bill'].isin([21.01, 23.68, 24.59])]
                    for column_mg in monitoring_detail[db][tb][scv][time_p].keys():
                        for value_s in monitoring_detail[db][tb][scv][time_p][column_mg].keys():
                            value = value_s.split(',')
                            ratio = data_df[data_df[column_mg].isin(value)]['_id'].count()/data_df['_id'].count()
                            print(scv,value,ratio)
                            for compare in monitoring_detail[db][tb][scv][time_p][column_mg][value_s].keys():
                                functions = Functions().ALIAS[compare]
                                threshold = float(monitoring_detail[db][tb][scv][time_p][column_mg][value_s][compare])
                                mt_result = locals()[functions](ratio,threshold)
                                mt_results = mt_result[0]
                                # print(mt_result)
                                if mt_results:
                                    email_open = email_open + 1
                                    mail_str = mail_str + mail_add.format(time_p,platform_scv[scv],column_mg,value,str(ratio*100)+"%",mt_result[1],str(threshold*100)+"%")
                                else:
                                    pass
    except BaseException as e:
        admin_str3 = mail_admin.format("查询计算",e)
        e_mail(my_admin,admin_str3)


max_time = datetime.datetime.now().strftime("%Y-%m-%d") + " 22:00:00"
max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')

min_time = datetime.datetime.now().strftime("%Y-%m-%d") + " 09:00:00"
min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')

# 每晚十点后 停止发送告警邮件
if datetime.datetime.now() >= min_time and datetime.datetime.now() <= max_time and email_open > 0 :
    ret = e_mail(my_user,mail_str)
    if ret :
        print("告警邮件发送成功!")
    else:
        print("告警邮件发送异常!")
else:
    print("未告警或者超出邮件发送时间")
print("监控结束",datetime.datetime.now())