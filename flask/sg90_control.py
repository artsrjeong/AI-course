from flask import Flask, request, render_template
#import RPi.GPIO as GPIO
import time

servoPin = 2
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3
cur_pos = 90

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(servoPin, GPIO.OUT)

#servo = GPIO.PWM(servoPin, 50)
#servo.start(0)

app = Flask(__name__)

def servo_control(degree, delay):
  if degree > 180:
    degree = 180

  duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
  # print("Degree: {} to {}(Duty)".format(degree, duty))
  # servo.ChangeDutyCycle(duty)
  time.sleep(delay)
 # servo.ChangeDutyCycle(0)


@app.route('/sg90_control')
def sg90_control():
  cur_pos = 90
  servo_control(cur_pos, 0.1)
  return render_template('sg90_control.html')
	
@app.route('/sg90_control_act', methods=['GET'])
def sg90_control_act(): 
  if request.method == 'GET':
    global cur_pos
    degree = ''
    servo = request.args["servo"]
	
    if servo == 'L':
      cur_pos = cur_pos - 10
      if cur_pos < 0:
        cur_pos = 0	
    else:
      cur_pos = cur_pos + 10 
      if cur_pos > 180: 
        cur_pos = 180
		
    servo_control(cur_pos, 0.1)	    
    return render_template('sg90_control.html', degree=cur_pos)	

if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0') 
