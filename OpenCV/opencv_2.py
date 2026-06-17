import cv2

img = cv2.imread("./OpenCV/img_1.png", cv2.IMREAD_COLOR)

if img is not None:
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이미지를 회색조로 변경

  img_resize = cv2.resize(img, (600, 400))  # 이미지 사이즈 변경
  gray_resize = cv2.resize(gray, (600, 400))  # 이미지 사이즈 변경

  cv2.imshow("img_resize", img_resize)
  cv2.imshow("gray_resize", gray_resize)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
else:
    print("Image file not found")
