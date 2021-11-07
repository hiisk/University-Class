# 학번, 이름 화면에 출력
id = '20187100'
name = '강현석'
print(id, name)

# 구글 드라이브 사용 권한 설정
from google.colab import drive
drive.mount('/gdrive')

# 과제 파일 저장한 폴더 위치 설정 (폴더 위치에 맞춰서 변경할 것)
import os
colab_path = "/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/CV_과제4"  # 경로예시: 구글드라이브/내 드라이브/Colab Notebooks/과제폴더명

"""# OpenCV 실습내용

## 색공간
"""

# OpenCV 패키지 사용
# !pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow  # colab 용 cv2.imshow
import matplotlib.pyplot as plt  # 시각화 도구

img = cv2.imread(os.path.join(colab_path, 'lena.jpg'), cv2.IMREAD_COLOR)  # 기본 BGR 채널로 이미지 읽기

img.shape  # numpy ndarray: (행, 열, 채널)

img.size  # 행 * 열 * 채널

img.dtype  # uint8: 정수형 [0, 255]

plt.imshow(img)  # OpenCV 기본 컬러순서: B, G, R

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 색공간변환: BGR -> RGB
# img_rgb = img[:, :, [2, 1, 0]]  # BGR -> RGB
# cv2_imshow(img_rgb)
plt.imshow(img_rgb)

img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 색공간변환: BGR -> YCrCb
img_y = img_ycrcb[:, :, 0]  # Y 채널 추출
plt.imshow(img_y, cmap='gray')  # grayscale로 plot

img_cr = img_ycrcb[:, :, 1]  # Cr 채널 추출
plt.imshow(img_cr, cmap='gray')  # grayscale로 plot

img_cb = img_ycrcb[:, :, 2]  # Cb 채널 추출
plt.imshow(img_cb, cmap='gray')  # grayscale로 plot

img_gray = cv2.imread(os.path.join(colab_path, 'lena.jpg'), cv2.IMREAD_GRAYSCALE)  # grayscale로 파일 읽기

img_gray.shape  # grayscale 1채널

img_gray.size  # 행 * 열 * 채널

img_gray.dtype  # uint8: 정수형 [0, 255]

"""## 이미지 읽기, 쓰기"""

img_rgb_crop = img_rgb[75:150, 150:250, :]  # 얼굴 절삭
plt.imshow(img_rgb_crop)

cv2.imwrite(os.path.join(colab_path, 'lena_face.png'), img_rgb_crop)  # png 파일 저장

img_sample = cv2.imread(os.path.join(colab_path, 'lena_face.png'), cv2.IMREAD_COLOR)  # png 파일 읽기
plt.imshow(img_sample)

cv2.imwrite(os.path.join(colab_path, 'lena_face.jpg'), img_rgb_crop)  # jpg 파일 저장

img_sample = cv2.imread(os.path.join(colab_path, 'lena_face.jpg'), cv2.IMREAD_COLOR)  # jpg 파일 읽기
plt.imshow(img_sample)

"""## 영상처리"""

import numpy as np

img = cv2.imread(os.path.join(colab_path, "lena.jpg"), cv2.IMREAD_COLOR)
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(imgGray, cmap='gray')  # grayscale로 plot

# GaussianBlur: 흐림 효과
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
plt.imshow(imgBlur, cmap='gray')

# Canny Edge Detection: 경계선 추출
imgCanny = cv2.Canny(imgGray, 150, 200)
plt.imshow(imgCanny, cmap='gray')

# 이미지 크기변화
img = cv2.imread(os.path.join(colab_path, "lena.jpg"), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_rgb.shape)

imgResize = cv2.resize(img_rgb, (800, 450))  # (열, 행)
print(imgResize.shape)

plt.imshow(img_rgb)  # 원본 영상 plot

plt.imshow(imgResize)  # 크기변화한 영상 plot

"""## 얼굴 탐지"""

faceCascade= cv2.CascadeClassifier(os.path.join(colab_path, "haarcascade_frontalface_default.xml"))
img = cv2.imread(os.path.join(colab_path, 'lena.png'))
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(img_gray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(img_rgb)

"""## 차량 번호판 탐지"""

frameWidth = 640
frameHeight = 480
nPlateCascade = cv2.CascadeClassifier(os.path.join(colab_path, "haarcascade_russian_plate_number.xml"))
minArea = 200
color = (0,255,0)
img_list = ['p1.jpg', 'p2.jpg', 'p3.jpg']
count = 0

for curr_img in img_list:
    img = cv2.imread(os.path.join(colab_path, curr_img), cv2.IMREAD_COLOR)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,"Number Plate",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv2_imshow(imgRoi)

    cv2_imshow(img)  # Colab용 cv2.imshow함수(BGR -> RGB 변환 포함)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,
                    2,(0,0,255),2)
        cv2_imshow(img)
        cv2.waitKey(500)

"""## 참고자료: OpenCV Haar/cascade training 튜토리얼

https://darkpgmr.tistory.com/70

# 실습 과제

문제: many_faces.jpg 파일의 모든 얼굴을 탐지하여 plot하기
- 사람 얼굴을 red box로 표시하기
- 얼굴 탐지 코드 활용하기
- 모든 사람의 얼굴에 red box는 1개만 표시하기
"""

faceCascade= cv2.CascadeClassifier(os.path.join(colab_path, "haarcascade_frontalface_default.xml"))
img = cv2.imread(os.path.join(colab_path, 'many_faces.jpg'), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(img_gray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv2_imshow(img)

