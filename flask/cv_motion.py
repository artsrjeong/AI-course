from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

threshold_move = 50  # 달라진 픽셀 값 기준치 설정
diff_compare = 10  # 달라진 픽셀 갯수 기준치 설정
img_first = None
img_second = None
img_third = None
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 영상의 폭을 320 으로 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 영상의 높이를 240 으로 설정

ret, img_first = cap.read()  # 1번재 프레임 읽기
ret, img_second = cap.read()  # 2번재 프레임 읽기


def gen_frames():  
  while True:
    global img_first
    global img_second
    global img_third
    global threshold_move
    global diff_compare
	
    ret, img_third = cap.read()  # 3번째 프레임 읽기
    scr = img_third.copy()  # 화면에 다른점 표시할 이미지 백업

    # 그레이 스케일로 변경
    img_first_gray = cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
    img_second_gray = cv2.cvtColor(img_second, cv2.COLOR_BGR2GRAY)
    img_third_gray = cv2.cvtColor(img_third, cv2.COLOR_BGR2GRAY)

    # 이미지간의 차이점 계산
    diff_1 = cv2.absdiff(img_first_gray, img_second_gray)
    diff_2 = cv2.absdiff(img_second_gray, img_third_gray)

    # Threshold 적용
    ret, diff_1_thres = cv2.threshold(diff_1, threshold_move, 255, \
                                                    cv2.THRESH_BINARY)
    ret, diff_2_thres = cv2.threshold(diff_2, threshold_move, 255, \
                                                    cv2.THRESH_BINARY)
 # 1번째영상-2번째영상, 2번째영상-3번째영상 차이점
    diff = cv2.bitwise_and(diff_1_thres, diff_2_thres)

    # 차이가 발생한 픽셀이 갯수 판단 후 사각형 그리기
    diff_cnt = cv2.countNonZero(diff)
    if diff_cnt > diff_compare:
      nzero = np.nonzero(diff)  # 0이 아닌 픽셀의 좌표 얻기
      cv2.rectangle(scr, (min(nzero[1]), min(nzero[0])), \
                      (max(nzero[1]), max(nzero[0])), (0, 255, 0), 1)
      cv2.putText(scr, "Motion Detected", (10, 10), \
                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 0))

    # 다음 비교를 위해 영상 저장
    img_first = img_second
    img_second = img_third
	
    ret, buffer = cv2.imencode('.jpg', scr) # 카메라 영상을 jpg 이미지 파일로 변환
    frame = buffer.tobytes()  # 바이트 형식으로 리턴

    # concat frame one by one and show result
    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/cv_motion')
def cv_motion():
    # rendering webpage
    return render_template('cv_motion.html')
	
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; \
                                         boundary=frame')
if __name__ == '__main__':
    # defining server ip address and port
    app.run(debug=False, port=80, host='0.0.0.0') 

