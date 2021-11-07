print("강현석", "20187100")

"""구글 드라이브 연결"""

from google.colab import drive
drive.mount('/gdrive')

"""작업폴더 경로 설정"""

workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/DA_과제5"  # 과제 파일 업로드한 경로 반영
# 작업폴더 경로 참고: 작업폴더 하위에 data 폴더 생성하여 CSV 파일 읽기/쓰기 수행
# A = os.path.join(workspace_path, 'data/winequality-red.csv')
# A = '/gdrive/My Drive/Colab Notebooks/data/winequality-red.csv'

"""파이썬 패키지 불러오기"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, glm

"""# 와인 품질 예측하기

*   목표: 와인의 속성을 분석하여 품질 등급 예측
*   레드 와인과 화이트 와인 그룹의 품질에 대한 t-검증 수행
*   와인 속성을 독립 변수로, 품질 등급을 종속 변수로 선형 회귀 분석을 수행

### 데이터 불러오기
"""

red_df = pd.read_csv(os.path.join(workspace_path, 'data/winequality-red.csv'), sep=';', header=0, engine='python')
red_df.head()

white_df = pd.read_csv(os.path.join(workspace_path, 'data/winequality-white.csv'), sep=';', header=0, engine='python')
white_df.head()

red_df.to_csv(os.path.join(workspace_path, 'data/winequality-red2.csv'), index=False)  # CSV 파일 출력
white_df.to_csv(os.path.join(workspace_path, 'data/winequality-white2.csv'), index=False)

"""### 데이터 병합: 레드 와인과 화이트 와인 파일 합치기"""

red_df.insert(0, column='type', value='red')  # 0번째 인덱스에 'type' 속성 추가 ('red'로 채우기)
red_df.head()

white_df.insert(0, column='type', value='white')  # 0번째 인덱스에 'type' 속성 추가 ('white'로 채우기)
white_df.head()

white_df.shape

wine = pd.concat([red_df, white_df])  # red, white 데이터 병합
wine.shape

wine.to_csv(os.path.join(workspace_path, 'data/wine.csv'), index=False)

wine.head()  # 앞부분 데이터 샘플 보기

wine.tail()  # 뒷부분 데이터 샘플 보기

"""### 데이터 탐색"""

wine.info()  # 데이터 기본 정보 확인하기

wine.columns = wine.columns.str.replace(' ', '_')  # 속성(column) 이름 공란을 '_'로 변경
wine.head()

sorted(wine.quality.unique())  # 'quality' 속성값 중에서 유일한 값만 출력

wine.quality.value_counts()  # 'quality' 속성값 빈도수 출력

wine['quality'].value_counts()  # 'quality' 속성값 빈도수 출력

"""### 데이터 모델링

#### describe() 함수로 그룹 비교
"""

wine.groupby('type')['quality'].describe()  # 'type'으로 그룹 나누기 -> 그룹 'quality' 통계치 출력

red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']  # 'type'이 'red'인 샘플의 'quality' 속성값 추출
red_wine_quality

white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']  # 'type'이 'white'인 샘플의 'quality' 속성값 추출
white_wine_quality

"""#### t-검정과 회귀 분석으로 그룹 비교"""

stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var=False)
# tvalue: 표본평균사이의 차이 / 표준오차
# pvalue <= 유의수준(통상 0.05), 두 모집단의 평균 간에 유의미한 차이 존재 (귀무가설 기각)

Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'

wine_train = pd.concat([red_df[:-5], white_df[:-5]])  # 학습용 red, white 데이터 병합
wine_train.columns = wine_train.columns.str.replace(' ', '_')  # 속성(column) 이름 공란을 '_'로 변경
wine_train.head()

regression_result = ols(Rformula, data=wine_train).fit()  # 선형 회귀 모델 학습(fitting)

regression_result.summary()  # 선형 회귀 분석 통계값 출력

"""#### 회귀 분석 모델로 새로운 샘플의 품질 등급 예측"""

sample1 = wine[wine.columns.difference(['quality', 'type'])]  # 'quality', 'type' 속성 제외한 나머지 독립 변수 추출
sample1 = sample1[-5:][:]  # 검증용 마지막 5개 샘플만 추출
sample1

sample1_predict = regression_result.predict(sample1)  # 학습한 회귀 분석 모델로 sample1의 'quality' 예측

sample1_predict  # 예측한 'quality' 확인

wine[-5:][['quality', 'type']]  # 정답 'quality' 확인

data = {'fixed_acidity': [8.5, 8.1], 'volatile_acidity': [0.8, 0.5],
        'citric_acid': [0.3, 0.4], 'residual_sugar': [6.1, 5.8],
        'chlorides': [0.055, 0.04], 'free_sulfur_dioxide': [30.0, 31.0],
        'total_sulfur_dioxide': [98.0, 99], 'density': [0.996, 0.91],
        'pH': [3.25, 3.01], 'sulphates': [0.4, 0.35], 'alcohol': [9.0, 0.88]
        }  # 임의의 데이터 샘플 sample2 생성

sample2 = pd.DataFrame(data, columns=sample1.columns)
sample2

sample2_predict = regression_result.predict(sample2)  # 학습한 회귀 분석 모델로 sample2의 'quality' 예측
sample2_predict

"""### 결과 시각화

#### distplot(히스토그램+커널밀도추정) 그리기
"""

sns.set_style('dark')
sns.distplot(red_wine_quality, kde=True, color='red', label='red wine')  # red wine distplot 객체 생성
sns.distplot(white_wine_quality, kde=True, label='white wine')  # white wine distplot 객체 생성
plt.title('Quality of Wine Type')  # 차트 제목 설정
plt.legend()  # 차트 범례 설정
plt.show()  # 차트 보이기

"""# 타이타닉호 생존율 분석하기

*   목표: 타이타닉호 승객 변수를 분석하여 생존 유무 예측
*   생존과 가장 상관관계가 높은 변수는 무엇인지 분석
*   상관관계 분석을 위해 피어슨 상관 계수를 사용
*   변수 간의 상관관계를 시각화하여 분석

### 데이터 준비
"""

titanic = sns.load_dataset('titanic')  # seaborn 패키지의 예제 데이터셋 'titanic' 불러오기
titanic

titanic.isnull().sum()  # 속성 별로 null값 개수 세기

titanic['age'] = titanic['age'].fillna(titanic['age'].median())  # null값을 중간값(median)으로 채우기

"""### 데이터 탐색"""

titanic.info()  # 데이터 기본 정보 탐색

# 남성(male) 생존율 pie 차트 (18.9% = 남성생존자 / 남성수)
f, ax = plt.subplots(1, 2, figsize=(10, 5))
titanic['survived'][titanic['sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
# 여성(female) 생존율 pie 차트 (25.8% = 여성생존자 / 여성수)
titanic['survived'][titanic['sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)
ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (Female)')

sns.countplot('pclass', hue='survived', data=titanic)  # 빈도수 막대그래프
plt.title('Pclass vs Survived')
plt.show()

"""### 데이터 모델링"""

titanic_corr = titanic.corr(method='pearson')  # 피어슨 상관 계수를 적용하여 계산
titanic_corr

titanic_corr.to_csv(os.path.join(workspace_path, 'data/titanic_corr.csv'), index=False)  # 상관계수 CSV 파일 출력

titanic['survived'].corr(titanic['adult_male'])  # 'survived'와 'adult_male' 변수 간의 상관계수

titanic['survived'].corr(titanic['fare'])  # 'survived'와 'fare' 변수 간의 상관계수

"""### 결과 시각화"""

sns.pairplot(titanic, hue='survived')  # 산점도로 상관 분석 시각화
plt.show()

sns.catplot(x='pclass', y='survived', hue='sex', data=titanic, kind='point')  # 성별 그룹 별로 생존자 객실 등급과 생존 상관관계 시각화
plt.show()

"""나이 정보 정제"""

def category_age(x):
    result = 0
    if x < 10:
        result = 0
    elif x < 20:
        result = 1
    elif x < 30:
        result = 2
    elif x < 40:
        result = 3
    elif x < 50:
        result = 4
    elif x < 60:
        result = 5
    elif x < 70:
        result = 6
    else:
        result = 7
    return result

titanic['age2'] = titanic['age'].apply(category_age)
titanic['sex'] = titanic['sex'].map({'male': 1, 'female': 0})  # 성별 정보 정제
titanic['family'] = titanic['sibsp'] + titanic['parch'] + 1  # 가족정보 정제
titanic.to_csv(os.path.join(workspace_path, 'data/titanic3.csv'), index=False)  # 현재 데이터 CSV 파일 출력
titanic.head()

heatmap_data = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]
colormap = plt.cm.RdBu
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True,
            annot_kws={'size': 10})
plt.show()

"""# 실습 과제

#### 1. titanic_train 데이터로 회귀 분석 모델을 학습하여 titanic_val 데이터의 생존 유무를 예측하기
"""

titanic_train = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']][:800] # 학습데이터
titanic_val = titanic[['sex', 'age2', 'family', 'pclass', 'fare']][800:] # 검증데이터
titanic_val_survived = titanic['survived'][800:]  # 검증데이터 생존자 정답
Rformula = 'survived ~ sex + age2 + family + pclass + fare'

"""1번 답안"""

# 학습 코드 작성
regression_result = ols(Rformula, titanic_train).fit()

regression_result.summary()  # 선형 회귀 분석 통계값 출력

# 예측 코드 작성
titanic_predict = regression_result.predict(titanic_val)

titanic_predict  # survived 예측값

titanic_val_survived  # survived 정답

titanic.head()

"""#### 2. 타이타닉 탑승자 성별(남성, 여성) 각각 age2 정보를 distplot 하기"""

# 코드 작성

# titanic_man = titanic.loc[titanic['who'] == 'man', 'age2']
# titanic_woman = titanic.loc[titanic['who'] == 'woman', 'age2']
# sns.set_style('dark')
# sns.distplot(titanic_man, kde=True, color='red', label='male pclass')
# sns.distplot(titanic_woman, kde=True, label='female pclass')
# plt.title('pclass of sex')  # 차트 제목 설정
# plt.legend()  # 차트 범례 설정
# plt.show()  # 차트 보이기

print(titanic_train)
titanic_man = titanic_train.loc[titanic_train['sex'] == 1, 'age2']
titanic_woman = titanic_train.loc[titanic_train['sex'] == 0, 'age2']
sns.set_style('dark')
sns.distplot(titanic_man, kde=True, color='red', label='male pclass')
sns.distplot(titanic_woman, kde=True, label='female pclass')


plt.title('pclass of sex')  # 차트 제목 설정
plt.legend()  # 차트 범례 설정
plt.show()  # 차트 보이기

#남자가1 여자가0 기준 그래프

