
from flask import Flask, request, render_template

app = Flask(__name__)     

@app.route('/method_get', methods=['GET'])
def method_get(): 
  return render_template('method_get.html')

# <form method="get" action="/method_get_act"> 태그에 의해서 호출
@app.route('/method_get_act', methods=['GET'])
def method_get_act(): 
  if request.method == 'GET': 
    # method_get.html에서 요청한 “id” 읽기
    id = request.args["id"]
    # method_get.html에서 요청한 “password” 읽기
    password = request.args.get("password")
    # method_get.html로 id, password 인자 전달
    return render_template('method_get.html', id=id, password=password)
		
if __name__ == '__main__':
  app.run(debug=True, port=80, host='0.0.0.0') 
