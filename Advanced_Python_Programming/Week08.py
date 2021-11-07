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
model = lgbm.LGBMClassifier(n_estimators=500)
model.fit(train_x, train_y)

"""예측치를 구해주고"""

pred_y = model.predict(test)

pred_y

submission['voted']=pred_y

submission

"""제출!!"""

submission.to_csv(os.path.join(dataset_path, 'sample_submission.csv'))

