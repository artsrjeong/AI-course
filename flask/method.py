
# Flask, Render 패키지 사용
from flask import Flask, request, render_template

# 인스턴스(객체) 생성
app = Flask(__name__)

# URL “/method” 로 요청이 오면
@app.route('/method', methods=['GET'])
def method(): 
  if request.method == 'GET': 
    id = request.args["id"]  # URL 에 있는 “id”
    password = request.args.get("password") # URL 에 있는 “password”
    return "GET으로 전달된 데이터({}, {})".format(id, password)
		
if __name__ == '__main__':
  app.run(debug=True, port=80, host='0.0.0.0')
