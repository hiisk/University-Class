"""이 게시글은 초급자 분들에게 제출 절차만을 설명드리기 위한 코드공유 글입니다.
   창의적인 전처리와 EDA를 통해 효율적인 알고리즘을 만들어 주세요"""


import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


"""데이터를 불러옵니다"""

from google.colab import drive
import os
drive.mount('/gdrive', force_remount=True)
# 아래 경로를 과제파일 저장한 경로로 수정하기 바랍니다.
dataset_path = '/gdrive/My Drive/Colab Notebooks/AI_과제8'


# 경로 설정을 꼭 신경써주세요!!
train = pd.read_csv(os.path.join(dataset_path, 'train.csv'), index_col=0) # 인덱스 col=0도 check!
test = pd.read_csv(os.path.join(dataset_path, 'test_x.csv'), index_col=0)
submission = pd.read_csv(os.path.join(dataset_path, 'submission_form.csv'), index_col=0)  
drop_val = ['QaA', 'QbA', 'QbE', 'QcA', 'QcE', 'QdE', 'QeA','QeE',
       'QfA', 'QfE', 'QgA', 'QgE', 'QhA', 'QhE', 'QiA', 'QiE', 'QjA', 'QjE',
       'QkA', 'QkE', 'QlA', 'QlE', 'QmA', 'QmE', 'QnA', 'QnE', 'QoA', 'QoE',
       'QpA', 'QpE', 'QqA', 'QqE', 'QrA', 'QrE', 'QsA', 'QsE', 'QtA', 'QtE','tp01', 'tp02', 'tp03', 'tp04', 'tp05',
       'tp06', 'tp07', 'tp08', 'tp09', 'tp10', 'urban', 'wf_01',
       'wf_02', 'wf_03', 'wr_01', 'wr_02', 'wr_03', 'wr_04', 'wr_05', 'wr_06',
       'wr_07', 'wr_08', 'wr_09', 'wr_10', 'wr_11', 'wr_12', 'wr_13']
train = train.drop(drop_val, axis = 1)
test = test.drop(drop_val, axis = 1)
train.head()
test.head()
print('train : ',train.isnull().sum())
print('test : ',test.isnull().sum())
train_y = train['voted']
tar = [str(train_y.unique()[1]), str(train_y.unique()[0])]
count= [train_y.value_counts()[1], train_y.value_counts()[2]]
plt.title('Voted')


"""타겟인 'voted'변수를 train_y로 만들어줍니다"""

train_x = train.drop('voted', axis = 1)
train_y = train['voted']


"""빠른 제출 절차를 확인하기 위해 '문자열' 변수들을 '이산형'으로 변환해줍니다.

그렇지만 이렇게 해서는 좋은 점수를 얻기 힘들 것입니다.
"""


#age_group, gender, race, religion
label = LabelEncoder()
train['age_group_code'] = label.fit_transform(train['age_group'])
test['age_group_code'] = label.fit_transform(test['age_group'])
train['race_code'] = label.fit_transform(train['race'])
test['race_code'] = label.fit_transform(test['race'])
train['religion_code'] = label.fit_transform(train['religion'])
test['religion_code'] = label.fit_transform(test['religion'])
train


"""모델을 돌려줍니다"""

# lgbm으로 분석
drop_last = ['familysize', 'gender','hand','education', 'age_group', 'race', 'religion']
new_train= train.drop(drop_last, axis = 1)
new_train = new_train.drop(['voted'], axis = 1)
test = test.drop(drop_last, axis = 1)


model = lgbm.LGBMClassifier(n_estimators=500)
model.fit(new_train, train_y)
pred_y = model.predict(test)


pred_y


submission['voted']=pred_y


submission


"""제출!!"""

submission.to_csv(os.path.join(dataset_path, 'submission.csv'))


"""## 실습 과제
### 모델링 방법으로 baseline 보다 높은 점수 획득하기
### baseline score: 0.6801542453
"""


# 모델 정의
# 모델 fit(학습)
# 아래 코드는 수정하지 말것
submission['voted'] = model.predict(test)  # 모델 예측(추론)
print(submission)
submission.to_csv(os.path.join(dataset_path, 'submission.csv'))