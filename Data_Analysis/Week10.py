"""
### ★ "Duplicate names are not allowed" 에러가 발생하지 않도록 pandas 버전을 down grade하여 설치 하기
#### 방법1) Anaconda Prompt를 관리자 권한으로 실행하여, pip install pandas==0.24.2
##### >> [Anaconda Prompt] 메뉴에서 마우스 오른쪽 버튼을 클릭하고 [자세히]-[관리자 권한으로 실행]한 후에 명령어 입력
#### 방법2) 주피터 노트북을 관리자 권한으로 실행한 후에, !pip install pandas==0.24.2
##### >> [Jupyter Notebook] 메뉴에서 마우스 오른쪽 버튼을 클릭하고 [자세히]-[관리자 권한으로 실행] 한 후에 명령어 입력
"""

!pip install pandas==0.24.2

"""수강생분의 이름과 학번을 입력해주세요."""

print("강현석", "20187100")

"""구글 드라이브 연결"""

from google.colab import drive
drive.mount('/gdrive')

"""작업폴더 경로 설정"""

import os
workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/DA_과제10"  # 과제 파일 업로드한 경로 반영
data_path = os.path.join(workspace_path, 'data')  # 데이터 파일 경로 반영

# 작업폴더 경로 참고: 작업폴더 하위에 data 폴더 생성하여 CSV 파일 읽기/쓰기 수행
# A = os.path.join(workspace_path, 'data/winequality-red.csv')  # workspace_path 이용한 경로 설정 예시
# A = '/gdrive/My Drive/Colab Notebooks/data/winequality-red.csv'  # 절대 경로 설정 예시

"""# 1. 로지스틱 회귀 분석을 이용한 유방암 진단 프로젝트

### 1) 데이터 수집
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

b_cancer = load_breast_cancer()

"""### 2) 데이터 수집 및 탐색"""

print(b_cancer.DESCR)

b_cancer_df = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)

b_cancer_df['diagnosis']= b_cancer.target

b_cancer_df.head()

print('유방암 진단 데이터셋 크기 : ', b_cancer_df.shape)

b_cancer_df.info()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

b_cancer_scaled = scaler.fit_transform(b_cancer.data)

print(b_cancer.data[0])

print(b_cancer_scaled[0])

"""### 3) 분석 모델 구축 : 로지스틱 회귀를 이용한 이진 분류 모델"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# X, Y 설정하기
Y = b_cancer_df['diagnosis']
X = b_cancer_scaled

# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 로지스틱 회귀 분석 : (1)모델 생성
lr_b_cancer = LogisticRegression()

# 로지스틱 회귀 분석 : (2)모델 훈련
lr_b_cancer.fit(X_train, Y_train)

# 로지스틱 회귀 분석 : (3)평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr_b_cancer.predict(X_test)
Y_predict

"""### 4) 결과 분석"""

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 오차 행렬
confusion_matrix(Y_test, Y_predict)

acccuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f},  F1: {3:.3f}'.format(acccuracy,precision,recall,f1))

print('ROC_AUC: {0:.3f}'.format(roc_auc))

"""# 2. 결정 트리 분석을 이용한 사용자 움직임 분류 프로젝트

### 1) 데이터 수집

#### - https://archive.ics.uci.edu/에 접속하여, ‘human activity recognition’를 검색한다.
-‘UCI HAR Dataset’  압축파일을 다운받아서, 압축을 푼다.

### 2) 데이터 준비 및 탐색
"""

import numpy as np
import pandas as pd

pd.__version__

# 피처 이름 파일 읽어오기
feature_name_df = pd.read_csv(os.path.join(data_path, 'features.txt'), sep='\s+',  header=None, names=['index', 'feature_name'], engine='python')

feature_name_df.head()

feature_name_df.shape

# index 제거하고, feature_name만 리스트로 저장
feature_name = feature_name_df.iloc[:, 1].values.tolist()

feature_name[:5]

X_train = pd.read_csv(os.path.join(data_path, 'train/X_train.txt'), sep='\s+', names=feature_name, engine='python')
X_test = pd.read_csv(os.path.join(data_path, 'test/X_test.txt'), sep='\s+', names=feature_name, engine='python')

Y_train = pd.read_csv(os.path.join(data_path, 'train/Y_train.txt'), sep='\s+', header=None, names=['action'], engine='python')
Y_test = pd.read_csv(os.path.join(data_path, 'test/Y_test.txt'), sep='\s+', header=None, names=['action'], engine='python')

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

X_train.info()

X_train.head()

print(Y_train['action'].value_counts())

label_name_df = pd.read_csv(os.path.join(data_path, 'activity_labels.txt'), sep='\s+',  header=None, names=['index', 'label'], engine='python')

# index 제거하고, feature_name만 리스트로 저장
label_name = label_name_df.iloc[:, 1].values.tolist()

label_name

"""### 3) 모델 구축 : 결정트리모델"""

from sklearn.tree import DecisionTreeClassifier

# 결정 트리 분류 분석 : 1) 모델 생성
dt_HAR = DecisionTreeClassifier(random_state=156)

# 결정 트리 분류 분석 : 2) 모델 훈련
dt_HAR.fit(X_train, Y_train)

# 결정 트리 분류 분석 : 3) 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = dt_HAR.predict(X_test)
Y_predict

"""### 4) 결과 분석"""

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_predict)
print('결정 트리 예측 정확도 : {0:.4f}'.format(accuracy))

"""#### ** 성능 개선을 위해 최적 파라미터 값 찾기"""

print('결정 트리의 현재 하이퍼 파라미터 : \n', dt_HAR.get_params())

from sklearn.model_selection import GridSearchCV

"""#### 최적 파라미터 찾기: max_depth 조절"""

params = {
    'max_depth' : [ 6, 8, 10, 12, 16, 20, 24]
}

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',
                       cv=5, return_train_score=True)
grid_cv.fit(X_train , Y_train)

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']]

print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))

"""#### 최적 파라미터 찾기: max_depth, min_samples_split 조절"""

params = {
    'max_depth' : [ 8, 16, 20 ],
    'min_samples_split' : [ 8, 16, 24 ]
}

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',
                       cv=5, return_train_score=True)
grid_cv.fit(X_train , Y_train)

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth','param_min_samples_split', 'mean_test_score', 'mean_train_score']]

print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))

best_dt_HAR = grid_cv.best_estimator_
best_Y_predict = best_dt_HAR.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)

print('best 결정 트리 예측 정확도 : {0:.4f}'.format(best_accuracy))

"""#### **  중요 피처 확인하기"""

import seaborn as sns
import matplotlib.pyplot as plt

feature_importance_values = best_dt_HAR.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)

feature_top10 = feature_importance_values_s.sort_values(ascending=False)[:10]

plt.figure(figsize = (10, 5))
plt.title('Feature Top 10')
sns.barplot(x=feature_top10, y=feature_top10.index)
plt.show()

"""## 5) Graphviz를 사용한 결정트리 시각화"""

!pip install graphviz

from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성.
export_graphviz(best_dt_HAR, out_file="tree.dot", class_names=label_name , feature_names = feature_name, impurity=True, filled=True)

import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()

"""
첫번째 줄: 분류 기준
gini: 지니 불순도(엔트로피)
Samples: 분류한 데이터 개수
Value: 클래스별 데이터 개수
Class: 예측한 클래스: 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
 'SITTING', 'STANDING', 'LAYING' 순
"""
graphviz.Source(dot_graph)

"""# 실습 과제

### 1) 데이터 수집
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()  # 붓꽃 데이터셋 불러오기

"""### 2) 데이터 수집 및 탐색"""

print(iris.DESCR)

#*   Sepal Length: 꽃받침의 길이 정보
#*   Sepal Width: 꽃받침의 너비 정보
#*   Petal Length: 꽃잎의 길이 정보
#*   Petal Width: 꽃잎의 너비 정보


iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['diagnosis']= iris.target

iris_df.head()

print('붓꽃 판별 데이터셋 크기 : ', iris_df.shape)

iris_df.info()

from sklearn.model_selection import train_test_split

# X, Y 설정하기
Y = iris_df['diagnosis']
X = iris_df.drop(columns=['diagnosis'])

Y.shape, X.shape

# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

"""### 3) 모델 구축 : 결정트리모델"""

from sklearn.tree import DecisionTreeClassifier

# 결정 트리 분류 분석 : 1) 모델 생성
dt_HAR = DecisionTreeClassifier(random_state=156)

# 결정 트리 분류 분석 : 2) 모델 훈련
dt_HAR.fit(X_train, Y_train)

# 결정 트리 분류 분석 : 3) 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = dt_HAR.predict(X_test)

"""### 4) 결과 분석"""

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_predict)
print('결정 트리 예측 정확도 : {0:.4f}'.format(accuracy))

"""#### ** 성능 개선을 위해 최적 파라미터 값 찾기"""

print('결정 트리의 현재 하이퍼 파라미터 : \n', dt_HAR.get_params())

from sklearn.model_selection import GridSearchCV

"""#### 최적 파라미터 찾기: max_depth 조절
- max_depth: [1, 2, 3, 4, 5, 10, 20, 30, None]
"""

params = {
    'max_depth' : [ 1, 2, 3, 4, 5, 10, 20, 30, None]
}

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',
                       cv=5, return_train_score=True)
grid_cv.fit(X_train , Y_train)

"""GridSearchCV 결과 테이블 출력"""

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']]

"""최고 평균 정확도, 최적 하이퍼 파라미터 출력"""

print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))

"""#### 최적 파라미터 찾기: max_depth, min_samples_split 조절
- max_depth: [1, 2, 3, 4, 5]
- min_samples_split: [2, 4, 6, 8]
"""

params = {
    'max_depth' : [ 1, 2, 3, 4, 5 ],
    'min_samples_split' : [ 2, 4, 6, 8 ]
}

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',
                       cv=5, return_train_score=True)
grid_cv.fit(X_train , Y_train)

"""GridSearchCV 결과 테이블 출력"""

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth','param_min_samples_split', 'mean_test_score', 'mean_train_score']]

"""최고 평균 정확도, 최적 하이퍼 파라미터 출력"""

print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))

"""best 결정 트리 예측 정확도 출력"""

best_dt_iris = grid_cv.best_estimator_
best_Y_predict = best_dt_iris.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)

print('best 결정 트리 예측 정확도 : {0:.4f}'.format(best_accuracy))

"""#### **  중요 피처 확인하기"""

import seaborn as sns
import matplotlib.pyplot as plt

"""중요 피처값 계산(Top 4) 및 바그래프 출력"""

feature_importance_values = best_dt_iris.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)
feature_top4 = feature_importance_values_s.sort_values(ascending=False)[:4]
plt.figure(figsize = (10, 5))
plt.title('Feature Top 4')
sns.barplot(x=feature_top4, y=feature_top4.index)
plt.show()

"""## 5) Graphviz를 사용한 결정트리 시각화"""

!pip install graphviz

feature_name = ['sepal length', 'sepal width', 'petal length', 'petal width']

label_name = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']

from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성.
export_graphviz(best_dt_iris, out_file="tree.dot", class_names=label_name , feature_names = feature_name, impurity=True, filled=True)

import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()

"""
첫번째 줄: 분류 기준
gini: 지니 불순도(엔트로피)
Samples: 분류한 데이터 개수
Value: 클래스별 데이터 개수
Class: 예측한 클래스: 'Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica' 순
"""
graphviz.Source(dot_graph)

