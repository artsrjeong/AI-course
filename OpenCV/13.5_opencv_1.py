import cv2
import numpy as np

threshold_move = 50  # 모션이 감지될 Threshold 지정
diff_compare = 10  # 달라진 픽셀 갯수 기준치 설정
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 영상의 폭을 320 으로 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 영상의 높이를 240 으로 설정

ret, img_first = cap.read()  # 1번째 프레임 읽기
cv2.imshow('scr', img_first)
ret, img_second = cap.read()  # 2번째 프레임 읽기
cv2.imshow('scr', img_second)

while True:
    ret, img_third = cap.read()  # 3번째 프레임 읽기
    scr = img_third.copy()  # 모션감지 표시용 이미지 백업

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

    # 모션 감지된 데이터 판단
    diff_cnt = cv2.countNonZero(diff)
    if diff_cnt > diff_compare:
        nzero = np.nonzero(diff)  # 0이 아닌 픽셀의 좌표 얻기
        cv2.rectangle(scr, (min(nzero[1]), min(nzero[0])), \
                      (max(nzero[1]), max(nzero[0])), (0, 255, 0), 1)
        cv2.putText(scr, "Motion Detected", (10, 10), \
                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 0))
    cv2.imshow('scr', scr)
    # 다음 비교를 위해 영상 저장
    img_first = img_second
    img_second = img_third

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
