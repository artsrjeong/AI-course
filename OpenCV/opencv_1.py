import cv2

img_file = "./OpenCV/img_1.png" # 이미지 경로
img = cv2.imread(img_file)    # 이미지 읽기

if img is not None:
  img_resize = cv2.resize(img, (960, 540))  # 이미지 사이즈 변경
  cv2.imshow("IMG", img_resize)      # 이미지를 표시
  cv2.waitKey()               # 키가 입력될 때 까지 대기
  cv2.destroyAllWindows()     # 모든 창 닫기
else:
    print("Image file not found")
