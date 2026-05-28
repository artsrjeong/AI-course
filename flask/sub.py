
from flask import Flask

app = Flask(__name__)     

@app.route('/')                   
def hello():
    return 'Hello Flask'

@app.route('/sub1')                   
def sub1():
    return 'SUB1 Page'

@app.route('/sub2')                   
def sub2():
    return 'SUB2 Page'

if __name__ == '__main__':
    app.run(debug=True, port=80, 
                       host='0.0.0.0') 
