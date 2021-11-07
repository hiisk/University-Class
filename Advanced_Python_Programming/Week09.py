"""

창의적인 전처리와 EDA를 통해 효율적인 알고리즘을 만들어 주세요
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, accuracy_score

"""데이터를 불러옵니다"""

from google.colab import drive
import os
drive.mount('/gdrive', force_remount=True)

# 아래 경로를 과제파일 저장한 경로로 수정하기 바랍니다.
dataset_path = '/gdrive/My Drive/Colab Notebooks/dacon/심리성향예측_AI경진대회'

# 경로 설정을 꼭 신경써주세요!!
train = pd.read_csv(os.path.join(dataset_path, 'train.csv'), index_col=0) # 인덱스 col=0도 check!
test = pd.read_csv(os.path.join(dataset_path, 'test_x.csv'), index_col=0)
submission = pd.read_csv(os.path.join(dataset_path, 'submission_form.csv'), index_col=0)

train

test

"""타겟인 'voted'변수를 train_y로 만들어줍니다"""

train_x = train.drop('voted', axis = 1)
train_y = train['voted']

"""빠른 제출 절차를 확인하기 위해 '문자열' 변수들을 '이산형'으로 변환해줍니다.

그렇지만 이렇게 해서는 좋은 점수를 얻기 힘들 것입니다.
"""

#age_group, gender, race, religion
train_x['age_group']=1
train_x['gender']=1
train_x['race']=1
train_x['religion']=1

test['age_group']=1
test['gender']=1
test['race']=1
test['religion']=1

"""모델을 돌려줍니다"""

# lgbm으로 분석
model = lgbm.LGBMClassifier(n_estimators=500, random_state=42)
model.fit(train_x, train_y)

pred_y = model.predict(test)

pred_y

submission['voted']=pred_y

submission

"""제출!!"""

submission.to_csv(os.path.join(dataset_path, 'submission.csv'))

"""## 실습 과제
### 1. 기본 문제: 모델링 방법으로 baseline 보다 높은 점수 획득하기
##### (기본 점수 10점 획득 조건: 획득한 점수 > baseline 점수)
### 2. 심화 문제: EDA + Feature Engineering 으로 max(baseline 점수, 1번 점수) 보다 높은 점수 획득하기 
##### (가산점 10점 획득 조건: 획득한 점수 > max(baseline 점수, 1번 점수)) 
### * baseline score: 0.6801542453
### * 유의사항: 데이콘 제출 제한 하루 3회
### * 코랩의 cross validaiton 상에서 높은 점수를 획득 -> 테스트셋에 대한 예측값 생성 -> 데이콘 제출

### 1. 기본 문제: 모델링 방법으로 baseline 보다 높은 점수 획득하기 (달성 시 10점)
"""

# 모델 정의
# 모델 fit(학습)

"""### 2. 심화 문제: EDA + Feature Engineering 으로 baseline(혹은 1번 점수) 보다 높은 점수 획득하기 (달성 시 10점)"""

# EDA(Exploratory Data Analysis, 탐색적 데이터분석) 수행
# Feature Engineering(특징 공학) 수행

"""### submission.csv 생성 부분 (수정하지 말 것)"""

# 아래 코드는 수정하지 말것
submission['voted'] = model.predict(test)  # 모델 예측(추론), 미리 최고 성능 모델을 변수 model에 할당시킬 것
print(submission)
submission.to_csv(os.path.join(dataset_path, 'submission.csv'))