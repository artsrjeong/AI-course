from flask import Flask,request, render_template
import pymysql

db = None 
cur = None 
app = Flask(__name__)

@app.route('/lm35_service')                   
def lm35_service():
  # 접속정보
  db = pymysql.connect(host='172.26.227.149', user='root', password='jung7574',db='mysql', charset='utf8')  
  cur = db.cursor() # 커서생성 
  sql = "SELECT DATATIME, TEMP FROM temperature ORDER BY DATATIME  ASC LIMIT 50" 
  
  # 실행할 sql문 
  cur.execute(sql)  
  result = cur.fetchall()
	
  db.close() # 종료
  return render_template("lm35_service.html", result=result)
  
if __name__ == '__main__':
  app.run(debug=True, port=80, host='0.0.0.0')
