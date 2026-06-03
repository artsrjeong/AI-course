
import pymysql    # pymysql 임포트

# 전역변수 선언부 
db = None 
cur = None 

# 접속정보
db = pymysql.connect(host='172.26.227.149', user='root', password='jung7574', db='mysql', charset='utf8')  

try:
  cur = db.cursor() # 커서생성 
  sql = "SELECT DATATIME, TEMP FROM temperature" 

  # 실행할 sql문 
  cur.execute(sql)
  
  result = cur.fetchall()
  for row in result:
    print(row[0], '|', row[1])

finally:
  db.close() # 종료
