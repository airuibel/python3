#-*- encoding:utf-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from impala.dbapi import connect
import datetime
import pandas as pds

my_sender = 'welabreport@wolaidai.com'
my_pass = 'Qv@xYZ9Ed%Y*JxH8'
my_user = ['kaka.yao@wolaidai.com']
my_user = ','.join(my_user)

# smtp.qq.com
# msg.as_string()

def mail(to_email,em_text):
    ret = True
    try:
        msg = MIMEText(em_text +'######表中配置时间的最大值非当前日期，需要注意', 'plain', 'utf-8')
        msg['From'] = formataddr(["WeLab Report<welabreport@wolaidai.com>", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        # msg['To'] = formataddr(["测试收件人", to_email])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['to'] = to_email
        msg['Subject'] = "任务监控"  # 邮件的主题，也可以说是标题

        server = smtplib.SMTP_SSL("smtp.exmail.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender, to_email.split(','), msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()
    except Exception:
        ret = False
    return ret

# ret = mail(my_user)
# if ret :
#     print('成功')
# else:
#     print('失败')


def conn(to_host,to_port,to_database):
    con = connect(host=to_host,port=to_port,database=to_database)
    cur = con.cursor()
    return cur

cur = conn('172.20.12.8',21050,'weplay_tmp')
sql_col = 'SELECT MAX(to_date(%s)) FROM %s WHERE %s is not null;'
sql_source  = 'SELECT * FROM  weplay_backup.kakatest ;'

cur.execute(sql_source)
data = cur.fetchall()
# data_dom = pds.DataFrame(data)
# print(data_dom)
# print(data)
# print(len(data))

n_time = datetime.datetime.now().date()
t_time = n_time.isoformat()

etl_job = []
# etl_job.append(['1','2','3'])
# etl_job.append(['3','4','5'])
# print(pds.DataFrame(etl_job))

for i in range(len(data)):
    ll = []
    for x in range(3):
        datas = data[i][x]
        # print(datas)
        ll.append(datas)
    # print(ll)
    sql_file = sql_col % (ll[1],ll[0],ll[1])
    print(sql_file)
    cur.execute(sql_file)
    data_s = cur.fetchall()[0][0]
    print(data_s)
    if data_s == t_time:
        print('ETL正常')
    else:
        etl_job.append(ll[0])
        print('ETL正常')

if len(etl_job) > 0:
    print("发送")
    etl_job_s =','.join(etl_job)
    ret = mail(my_user, etl_job_s)
    if ret:
        print('ETL异常，邮件发送正常')
    else:
        print('ETL异常，邮件异常')
else:
    print("正常")

