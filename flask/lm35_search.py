from flask import Flask,request, render_template
import pymysql


db = None 
cur = None 
app = Flask(__name__)

def select(query):
  # 접속정보
  db = pymysql.connect(host='172.26.227.149', user='root', password='jung7574',   
                                                   db='mysql', charset='utf8')  
  # 커서생성
  cur = db.cursor() 

  # 실행할 sql문 
  cur.execute(query)
  result = cur.fetchall()
  db.close() # 종료
  return result
 
@app.route('/lm35_search')                   
def lm35_search():
  sql = "SELECT DATATIME, TEMP FROM temperature ORDER BY DATATIME ASC LIMIT 100" 
  result = select(sql)  
  return render_template("lm35_search.html", result=result)
  
@app.route('/lm35_search_act', methods=['GET', 'POST'])                   
def lm35_search_act():
  if request.method == 'POST': 
    start = request.form["start"]
    end = request.form["end"]  
    sql = "SELECT DATATIME, TEMP FROM temperature WHERE DATATIME >= '%s' AND DATATIME <= '%s' ORDER BY DATATIME ASC LIMIT 100" % (start, end)
    result = select(sql)  
    return render_template("lm35_search.html", result=result) 
  
if __name__ == '__main__':
  app.run(debug=True, port=80, host='0.0.0.0')
