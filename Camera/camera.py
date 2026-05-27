import cv2

# DroidCam 가상 웹캠 장치 인덱스 지정 (기본은 0, 안 나오면 1이나 2로 변경)
CAMERA_INDEX = 0

# VideoCapture 객체 생성
cap = cv2.VideoCapture(CAMERA_INDEX)

# 해상도 설정 (DroidCam 무료 버전은 기본 480p, 유료는 720p/1080p 지원)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("스마트폰 카메라(DroidCam)를 열 수 없습니다. 인덱스를 확인하세요.")
    exit()

print("카메라 연결 성공! 'q'를 누르면 종료됩니다.")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 화면에 출력
    cv2.imshow('Galaxy WebCam Test', frame)

    # 'q' 키를 누르면 루프 탈출
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
