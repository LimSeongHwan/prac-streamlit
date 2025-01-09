import pymysql

def connect_db(query):
    conn = pymysql.connect(host='127.0.0.1', user='root', password='root', db='dbaas_be', charset='utf8');
    cur = conn.cursor()
    cur.execute(query)
    conn.close()
    return cur.fetchall()
