# 수강생 분의 이름과 학번을 입력해주세요.
"""

print("강현석", "20187100")

"""# Getting Started with pandas"""

import pandas as pd

from pandas import Series, DataFrame

import numpy as np

"""## Introduction to pandas Data Structures

### Series
"""

obj = pd.Series([4, 7, -5, 3])
obj

print(obj.values)
print(obj.index)  # like range(4)

obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2

obj2.index

print(obj2['a'])
obj2['d'] = 6
print(obj2[['c', 'a', 'd']])

print(obj2 > 0)
obj2[obj2 > 0]

print(obj2 * 2)
print(np.exp(obj2))

print('b' in obj2)
print('e' in obj2)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4

pd.isnull(obj4)

pd.notnull(obj4)

obj4.isnull()

"""### DataFrame"""

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)

frame

frame.head()

frame.tail()

pd.DataFrame(data, columns=['year', 'state', 'pop'])

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
print(frame2)
print(frame2.columns)
print(frame2.index)
# 리스트(list): 새로운 라벨 인덱싱하면 오류 발생
# 딕셔너리(dictionary): 새로운 라벨 인덱싱하면 새로운 라벨 생성

frame2['state']

frame2.year

frame2.loc['three']  # loc: location 의미, [행, 열] 순으로 라벨 인덱싱할 때 사용

frame2.loc['three', 'state']  # loc: location 의미, [행, 열] 순으로 라벨 인덱싱할 때 사용

frame2['debt'] = 16.5
print(frame2)
frame2['debt'] = np.arange(6.)
print(frame2)

val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2

frame2['eastern'] = frame2.state == 'Ohio'
frame2

del frame2['eastern']
frame2.columns

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = pd.DataFrame(pop)
print(frame3.index)
frame3

frame3.T

pd.DataFrame(pop, index=[2001, 2002, 2003])

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)

"""## Essential Functionality

### Dropping Entries from an Axis
"""

obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
print(obj)
print(obj.drop(['d', 'c']))  # drop: index 접근하여 값 삭제

obj = pd.DataFrame(np.random.rand(5, 4), index=['a', 'b', 'c', 'd', 'e'])
print(obj)
print(obj.drop(['d', 'c']))  # drop: index 접근하여 값 삭제

"""### Indexing, Selection, and Filtering

#### Selection with loc and iloc
"""

data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])

print(data)
data.loc['Colorado', ['two', 'three']]  # loc 메소드는 라벨로 인덱싱 [행, 열] 순

print(data.iloc[2, [3, 0, 1]])  # iloc 메소드는 숫자(순서)로 인덱싱 [행, 열] 순
print(data.iloc[2])
print(data.iloc[[1, 2], [3, 0, 1]])

print(data)
print('\n')
print(data.loc[:'Utah', 'two'])
print('\n')
print(data.iloc[:, :3][data.three > 5])

"""### Sorting"""

obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()

obj.sort_index(ascending=False)  # ascending: 오름차순, false 설정 시 내림차순

obj = pd.Series([4, 7, -3, 2])
obj.sort_values()

"""### Axis Indexes with Duplicate Labels"""

obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj

obj.index.is_unique

df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
print(df)
df.loc['b']

"""## Summarizing and Computing Descriptive Statistics"""

df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df

df.sum()

df.idxmax()  # max 값 인덱스 찾기

df.cumsum()  # 누적합 계산

df.describe()  # DataFrame 주요 통계지표 출력

obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
print(obj)
obj.describe()

"""### Pandas Profiling (판다스 프로파일링)
##### 자주 사용하는 분석 정보를 report 로 생성하는 도구
##### [GitHub 링크](https://github.com/pandas-profiling/pandas-profiling)
"""

import pandas as pd
data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4],
                     'Qu2': [2, 3, 1, 2, 3],
                     'Qu3': [1, 5, 2, 4, 4]})
data

!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

from pandas_profiling import ProfileReport

profile = ProfileReport(data, title="Pandas Profiling Report")
profile

from google.colab import drive
drive.mount('/gdrive')

profile.to_file("/gdrive/My Drive/Colab Notebooks/pandas_report.html")
profile.to_file("/gdrive/My Drive/Colab Notebooks/pandas_report.json")

from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file

if __name__ == "__main__":
    file_name = cache_file(
        "titanic.csv",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    )

    df = pd.read_csv(file_name)

    profile = ProfileReport(df, title="Titanic Dataset", explorative=True)
    profile.to_file("/gdrive/My Drive/Colab Notebooks/titanic_report.html")

