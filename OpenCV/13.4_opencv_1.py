import numpy as np
import cv2
CAMERA_INDEX = 0
face_cascade = cv2.CascadeClassifier  \
 (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade =  \
 cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(CAMERA_INDEX) # 첫번째 카메라 영상
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 카메라 영상 넓이
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 카메라 영상 높이
while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)  # ①②③④⑤
    print("Number of faces detected: " + str(len(faces)))

    for (x, y, w, h) in faces:   # ②
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1) # ③
        roi_gray = gray[y:y + h, x:x + w] # ④
        roi_color = img[y:y + h, x:x + w] # ⑤
        eyes = eye_cascade.detectMultiScale(roi_gray) # ⑥
        for (ex, ey, ew, eh) in eyes:  # ⑦
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1) # ⑧

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
