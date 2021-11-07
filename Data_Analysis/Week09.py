print("강현석", "20187100")

"""구글 드라이브 연결"""

from google.colab import drive
drive.mount('/gdrive')

"""작업폴더 경로 설정"""

import os
workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/DA_과제9"  # 과제 파일 업로드한 경로 반영
data_path = os.path.join(workspace_path, 'data')  # 데이터 파일 경로 반영

# 작업폴더 경로 참고: 작업폴더 하위에 data 폴더 생성하여 CSV 파일 읽기/쓰기 수행
# A = os.path.join(workspace_path, 'data/winequality-red.csv')  # workspace_path 이용한 경로 설정 예시
# A = '/gdrive/My Drive/Colab Notebooks/data/winequality-red.csv'  # 절대 경로 설정 예시

"""# 1. 주택가격 회귀 분석"""

# 향후 버전 업에 대한 경고 메시지 출력 안하기 
import warnings

warnings.filterwarnings(action='ignore')

"""### - 머신러닝 패키지 sklearn 설치"""

!pip install sklearn

"""## 1) 데이터 수집"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
boston =  load_boston()  # 보스톤 주택가격 예측 데이터셋 로드

"""## 2) 데이터 준비 및 탐색"""

print(boston.DESCR)  # 데이터셋 설명 출력

boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df.head()

boston_df['PRICE'] = boston.target
boston_df.head()

print('보스톤 주택 가격 데이터셋 크기 : ', boston_df.shape)

boston_df.info()  # 데이터 정보 확인 (결측값 포함)

"""## 3) 분석 모델 구축"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# X, Y 분할하기
Y = boston_df['PRICE']
X = boston_df.drop(['PRICE'], axis=1, inplace=False)

# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=156)

# 선형회귀분석 : 모델 생성
lr = LinearRegression()

# 선형회귀분석 : 모델 훈련
lr.fit(X_train, Y_train)

# 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr.predict(X_test)

"""## 4) 결과 분석 및 시각화"""

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))

print('Y 절편 값: ', lr.intercept_)
print('회귀 계수 값: ', np.round(lr.coef_, 1))

coef = pd.Series(data = np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending = False)

"""## - 회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기"""

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=5)

x_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='PRICE', data=boston_df, ax=axs[row][col])

"""# 2. 자동차 연비 예측 분석"""

# 향후 버전 업에 대한 경고 메시지 출력 안하기
import warnings

warnings.filterwarnings(action='ignore')

"""## 1) 데이터 수집"""

import numpy as np
import pandas as pd
import os

data_df = pd.read_csv(os.path.join(data_path, 'auto-mpg.csv'), header=0, engine='python')
ori_data_df = data_df.copy()

"""## 2) 데이터 준비 및 탐색"""

print(' 데이터셋 크기 : ', data_df.shape)

data_df.head()

"""#### - 분석하지 않을 변수 제외하기"""

data_df = data_df.drop(['car_name', 'origin', 'horsepower'], axis=1, inplace=False)

print(' 데이터세트 크기 : ', data_df.shape)

data_df.head()

data_df.info()

"""## 3) 분석 모델 구축"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# X, Y 분할하기

Y = data_df['mpg']
X = data_df.drop(['mpg'], axis=1, inplace=False)

# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 선형회귀분석 : 모델 생성
lr = LinearRegression()

# 선형회귀분석 : 모델 훈련
lr.fit(X_train, Y_train)

# 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr.predict(X_test)

"""## 4) 결과 분석 및 시각화"""

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))

print('Y 절편 값: ',  np.round(lr.intercept_, 2))
print('회귀 계수 값: ', np.round(lr.coef_, 2))

coef = pd.Series(data=np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending=False)

"""### - 회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기"""

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)

x_features = ['model_year', 'acceleration', 'displacement', 'weight', 'cylinders']
plot_color = ['r', 'b', 'y', 'g', 'r']

for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='mpg', data=data_df, ax=axs[row][col], color=plot_color[i])

"""###   <<<< 연비 예측하기  >>>>"""

ori_data_df['car_name']  # 차량 목록 확인

sample_car_name = 'bmw 320i'  # 연비 예측할 차량 선택

"""BMW 320i 사진

"""

sample_car = ori_data_df[ori_data_df['car_name'] == 'bmw 320i']

cylinders_1 = sample_car['cylinders'].values[0]
displacement_1 = sample_car['displacement'].values[0]
weight_1 = sample_car['weight'].values[0]
acceleration_1 = sample_car['acceleration'].values[0]
model_year_1 = sample_car['model_year'].values[0]

mpg_predict = lr.predict([[cylinders_1, displacement_1, weight_1, acceleration_1 , model_year_1]])

print("이 자동차의 예상 연비(mpg)는 %.2f 입니다." %mpg_predict)

"""# 실습 과제

#### 사이킷런의 위스콘신 유방암 진단 데이터를 회귀분석+시각화하고 로지스틱 회귀를 이용하여 예측 모델 학습하기

- **위스콘신 유방암 진단 데이터셋 정보 링크**
 - https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- **사이킷런 load_breast_cancer() 정보**
 - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
- **로지스틱 회귀분석 사용법**
 - 모듈 임포트: from sklearn.linear_model import LogisticRegression
 - 모델 생성: lr = LogisticRegression()
- **Accuracy 성능지표**
 - 모듈 임포트: from sklearn.metrics import accuracy_score
 - 정확도 측정: accuracy = accuracy_score(Y_test, Y_predict)

유방암 진단 데이터셋 로드
"""

from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()  # 유방암 예측 데이터셋 로드

"""유방암 진단 데이터셋 설명 출력"""

print(breast_cancer.DESCR)  # 데이터셋 설명 출력

"""유방암 진단 데이터로 DataFrame 생성하기"""

breast_cancer_df = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
breast_cancer_df.head()

"""DataFrame에 정답('target') column 추가하기"""

breast_cancer_df['target'] = breast_cancer.target
breast_cancer_df.head()

"""유방암 진단 데이터셋 크기 출력"""

print('유방암 진단 데이터셋 크기 : ', breast_cancer_df.shape)

"""데이터 정보 확인(결측값 포함)"""

breast_cancer_df.info()

"""산점도 + 선형 회귀 그래프로 시각화하기"""

fig, axs = plt.subplots(figsize=(16, 40), ncols=3, nrows=11)

x_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error',
'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension', 'target']

for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='target', data=breast_cancer_df, ax=axs[row][col])

"""필요한 모듈 import"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
import os

"""결과 재현을 위한 설정"""

seed = 719
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

"""데이터셋에서 데이터(X), 정답(Y) 분할하기"""

Y = breast_cancer_df['target']
X = breast_cancer_df.drop(['target'], axis=1, inplace=False)

"""훈련용 데이터와 평가용 데이터 분할하기 (seed로 random_state 설정)"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = seed)

"""로지스틱 회귀 분석 모델 생성"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

"""로지스틱 회귀 분석 모델 훈련"""

lr.fit(X_train, Y_train)

"""로지스틱회귀분석 모델로 평가 데이터에 대한 예측 수행"""

Y_predict = lr.predict(X_test)
Y_predict

"""예측 정확도 계산"""

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)
print("Accuracy: %.4f"%accuracy)

