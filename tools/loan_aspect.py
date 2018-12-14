# -*- coding: UTF-8 -*-
import re
import commands
import datetime

bgn_date = (datetime.date.today() - datetime.timedelta(1)).isoformat()
end_date = (datetime.date.today() - datetime.timedelta(1)).isoformat()
#end_date = '2015-12-31'

fp = open('/disk1/bdl/kdm/etl_job/dpd_job/loan_aspect.sql')
content = fp.read()
fp.close

cmd_str = 'impala-shell --impalad=172.20.12.8:21000 --query="%s" --ldap --auth_creds_ok_in_clear --user=sichar.liu --ldap_password_cmd="echo -n 123456"'


def excute_sql(dateasp):
    sql_str = content
    sql_str = sql_str.replace('2017-12-31', dateasp.isoformat())
    sqlfile = '/disk1/bdl/kdm/etl_job/dpd_job/loan_aspect-%s.sql' % (dateasp)
    fp = open(sqlfile, 'w')
    fp.write(sql_str)
    fp.close
    result = commands.getstatusoutput(cmd_str % sql_str)
    print('################################ %s ################################' % result[0])
    print(result[1])


init_sql = 'TRUNCATE TABLE wp_calc.loan_aspect;'
result = commands.getstatusoutput(cmd_str % init_sql)
print('################################ %s ################################' % result[0])
print(result[1])

dateasp = datetime.datetime.strptime(bgn_date,'%Y-%m-%d').date()
while dateasp <= datetime.datetime.strptime(end_date,'%Y-%m-%d').date():
    if dateasp.isoweekday() == 7 or (dateasp + datetime.timedelta(1)).day == 1:
        excute_sql(dateasp)
    dateasp += datetime.timedelta(1)

