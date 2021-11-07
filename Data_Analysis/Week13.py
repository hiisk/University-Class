print("강현석", "20187100")

"""구글 드라이브 연결"""

from google.colab import drive
drive.mount('/gdrive')

"""작업폴더 경로 설정"""

import os
workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/DA_과제13"  # 과제 파일 업로드한 경로 반영
data_path = os.path.join(workspace_path, 'data')  # 데이터 파일 경로 반영

# 작업폴더 경로 참고: 작업폴더 하위에 data 폴더 생성하여 CSV 파일 읽기/쓰기 수행
# A = os.path.join(workspace_path, 'data/winequality-red.csv')  # workspace_path 이용한 경로 설정 예시
# A = '/gdrive/My Drive/Colab Notebooks/data/winequality-red.csv'  # 절대 경로 설정 예시

"""# 군집분석 : 타깃마케팅을 위한 K-평균 군집화

### 1) 데이터 수집
"""

import pandas as pd
import math

retail_df = pd.read_excel(os.path.join(data_path, 'Online_Retail.xlsx'))
retail_df.head()

retail_df.shape

"""### 2) 데이터 준비 및 탐색"""

retail_df.info()

# 오류 데이터 정제
retail_df = retail_df[retail_df['Quantity'] > 0]
retail_df = retail_df[retail_df['UnitPrice'] > 0]
retail_df = retail_df[retail_df['CustomerID'].notnull()]

# 'CustomerID' 자료형을 정수형으로 변환
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)

retail_df.info()
print(retail_df.isnull().sum())
print(retail_df.shape)

# 중복 레코드 제거
retail_df.drop_duplicates(inplace=True)

print(retail_df.shape) # 작업 확인용 출력

"""#### - 제품 수, 거래건 수, 고객 수 탐색"""

pd.DataFrame([{'Product':len(retail_df['StockCode'].value_counts()),
               'Transaction':len(retail_df['InvoiceNo'].value_counts()),
               'Customer':len(retail_df['CustomerID'].value_counts())}],
                          columns = ['Product', 'Transaction', 'Customer'],
                          index = ['counts'])

retail_df['Country'].value_counts()

# 주문금액 컬럼 추가
retail_df['SaleAmount'] = retail_df['UnitPrice'] * retail_df['Quantity']

retail_df.head() # 작업 확인용 출력

"""#### - 고객의 마지막 주문후 경과일(Elapsed Days), 주문횟수(Freq), 주문 총액(Total Amount) 구하기"""

aggregations = {
    'InvoiceNo':'count',
    'SaleAmount':'sum',
    'InvoiceDate':'max'
}

customer_df = retail_df.groupby('CustomerID').agg(aggregations)
customer_df = customer_df.reset_index()

customer_df.head()  # 작업 확인용 출력

# 컬럼이름 바꾸기
customer_df = customer_df.rename(columns = {'InvoiceNo':'Freq', 'InvoiceDate':'ElapsedDays'})

customer_df.head() # 작업 확인용 출력

"""#### - 마지막 구매후 경과일 계산하기"""

import datetime

customer_df['ElapsedDays'] = datetime.datetime(2011,12,10) - customer_df['ElapsedDays']

customer_df.head() # 작업 확인용 출력

customer_df['ElapsedDays'] = customer_df['ElapsedDays'].apply(lambda x: x.days+1)
customer_df.head() # 작업 확인용 출력

"""#### - 현재 데이터 값의 분포 확인하기"""

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
ax.boxplot([customer_df['Freq'], customer_df['SaleAmount'], customer_df['ElapsedDays']], sym='bo')
plt.xticks([1, 2, 3], ['Freq', 'SaleAmount','ElapsedDays' ])

plt.show()

"""#### - 데이터 값의 왜곡(치우침)을 줄이기 위한 작업 : 로그 함수로 분포 조정"""

import numpy as np

customer_df['Freq_log'] = np.log1p(customer_df['Freq'])
customer_df['SaleAmount_log'] = np.log1p(customer_df['SaleAmount'])
customer_df['ElapsedDays_log'] = np.log1p(customer_df['ElapsedDays'])

customer_df.head()  # 작업 확인용 출력

# 조정된 데이터 분포를 다시 박스플롯으로 확인하기
fig, ax = plt.subplots()
ax.boxplot([customer_df['Freq_log'], customer_df['SaleAmount_log'],customer_df['ElapsedDays_log']], sym='bo')
plt.xticks([1, 2, 3], ['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'])
plt.show()

"""### 3) 모델 구축 : K-평균 군집화 모델"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

X_features = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']].values

# 정규 분포로 다시 스케일링하기
from sklearn.preprocessing import StandardScaler

X_features_scaled = StandardScaler().fit_transform(X_features)

"""### - 최적의 k 찾기 (1) 엘보우 방법

- 왜곡: 클러스터의 중심점과 클러스터 내의 데이터 거리 차이의 제곱값의 합
- 엘보우 방법: 그래프가 꺾이는 지점인 k를 최적의 k로 선택
"""

distortions = []

for i in range(1, 11):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모델 생성
    kmeans_i.fit(X_features_scaled)   # 모델 훈련
    distortions.append(kmeans_i.inertia_)

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0) # 모델 생성

# 모델 학습과 결과 예측(클러스터 레이블 생성)
Y_labels = kmeans.fit_predict(X_features_scaled)

customer_df['ClusterLabel'] = Y_labels

customer_df.head()  # 작업 확인용 출력

"""## 4) 결과 분석 및 시각화

### - 최적의 k 찾기 (2) 실루엣 계수에 따른 각 클러스터의 비중 시각화 함수 정의
"""

from matplotlib import cm

def silhouetteViz(n_cluster, X_features):

    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    silhouette_values = silhouette_samples(X_features, Y_labels, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster):
        c_silhouettes = silhouette_values[Y_labels == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(float(c) / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)

    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.title('Number of Cluster : '+ str(n_cluster)+'\n' \
              + 'Silhouette Score : '+ str(round(silhouette_avg,3)))
    plt.yticks(y_ticks, range(n_cluster))
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()

"""### - 클러스터 수에 따른 클러스터 데이터 분포의 시각화 함수 정의"""

def clusterScatter(n_cluster, X_features):
    c_colors = []
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster) # 클러스터의 색상 설정
        c_colors.append(c_color)
        # 클러스터의 데이터 분포를 동그라미로 시각화
        plt.scatter(X_features[Y_labels == i,0], X_features[Y_labels == i,1],
                     marker='o', color=c_color, edgecolor='black', s=50,
                     label='cluster '+ str(i))

    # 각 클러스터의 중심점을 삼각형으로 표시
    for i in range(n_cluster):
        plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1],
                    marker='^', color=c_colors[i], edgecolor='w', s=200)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

"""실루엣 계수: -1에서 1 사이의 값을 가지며, 1에 가까울수록 좋은 군집화를 의미"""

silhouetteViz(3, X_features_scaled) # 클러스터 3개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

silhouetteViz(4, X_features_scaled)  # 클러스터 4개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

silhouetteViz(5, X_features_scaled) # 클러스터 5개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

silhouetteViz(6, X_features_scaled) # 클러스터 6개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

clusterScatter(3, X_features_scaled) # 클러스터 3개인 경우의 클러스터 데이터 분포 시각화

clusterScatter(4, X_features_scaled)  # 클러스터 4개인 경우의 클러스터 데이터 분포 시각화

clusterScatter(5, X_features_scaled)  # 클러스터 5개인 경우의 클러스터 데이터 분포 시각화

clusterScatter(6, X_features_scaled)  # 클러스터 6개인 경우의 클러스터 데이터 분포 시각화

"""### 결정된 k를 적용하여 최적의 K-means 모델 완성"""

best_cluster = 4

kmeans = KMeans(n_clusters=best_cluster, random_state=0)
Y_labels = kmeans.fit_predict(X_features_scaled)

customer_df['ClusterLabel'] = Y_labels

customer_df.head()   # 작업 확인용 출력

"""#### - ClusterLabel이 추가된 데이터를 파일로 저장"""

customer_df.to_csv(os.path.join(data_path, 'Online_Retail_Customer_Cluster.csv'))

"""## << 클러스터 분석하기 >>

### 1) 각 클러스터의 고객수
"""

customer_df.groupby('ClusterLabel')['CustomerID'].count()

"""### 2) 각 클러스터의 특징"""

customer_cluster_df = customer_df.drop(['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'],axis=1, inplace=False)

# 주문 1회당 평균 구매금액 : SaleAmountAvg
customer_cluster_df['SaleAmountAvg'] = customer_cluster_df['SaleAmount']/customer_cluster_df['Freq']

customer_cluster_df.head()

# 클러스터별 분석
customer_cluster_df.drop(['CustomerID'],axis=1, inplace=False).groupby('ClusterLabel').mean()

"""# 실습 과제

# K-평균으로 붓꽃 군집 분석하기

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

"""#### - 현재 데이터 값의 분포 확인하기: iris 데이터셋의 4가지 특성 박스플롯"""

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
ax.boxplot([iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], iris_df['petal length (cm)'], iris_df['petal width (cm)']], sym='bo')
plt.xticks([1, 2, 3, 4], ['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width'])

plt.show()

"""### 3) 모델 구축 : K-평균 군집화 모델"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

X_features = iris_df[['sepal length (cm)', 'sepal width (cm)',
                      'petal length (cm)', 'petal width (cm)']].values

# 정규 분포로 다시 스케일링하기
from sklearn.preprocessing import StandardScaler

X_features_scaled = StandardScaler().fit_transform(X_features)

"""### - 최적의 k 찾기 (1) 엘보우 방법"""

distortions = []

for i in range(1, 11):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모델 생성
    kmeans_i.fit(X_features_scaled)   # 모델 훈련
    distortions.append(kmeans_i.inertia_)

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0) # 모델 생성

# 모델 학습과 결과 예측(클러스터 레이블 생성)
Y_labels = kmeans.fit_predict(X_features_scaled)

iris_df['diagnosis'] = Y_labels

iris_df.head()  # 작업 확인용 출력

"""## 4) 결과 분석 및 시각화

### 실루엣 계수에 따른 각 클러스터의 비중 시각화
"""

silhouetteViz(2, X_features_scaled) # 클러스터 2개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

silhouetteViz(3, X_features_scaled) # 클러스터 3개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

silhouetteViz(4, X_features_scaled)  # 클러스터 4개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

clusterScatter(2, X_features_scaled) # 클러스터 2개인 경우의 클러스터 데이터 분포 시각화

clusterScatter(3, X_features_scaled) # 클러스터 3개인 경우의 클러스터 데이터 분포 시각화

clusterScatter(4, X_features_scaled)  # 클러스터 4개인 경우의 클러스터 데이터 분포 시각화

