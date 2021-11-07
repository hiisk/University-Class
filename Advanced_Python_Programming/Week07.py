# 수강생 분의 이름과 학번을 입력해주세요.
"""

print("강현석", "20187100")

# 구글 드라이브 사용 권한 설정
from google.colab import drive
drive.mount('/gdrive')

# examples 폴더 위치 설정 (폴더 위치에 맞춰서 변경할 것)
examples_path = "/gdrive/My Drive/Colab Notebooks/APP_과제7/examples"

"""# Data Loading, Storage, """

import numpy as np
import pandas as pd
import os
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

"""## Reading and Writing Data in Text Format"""

with open(os.path.join(examples_path, "ex1.csv"), 'r') as f:
  print(f.read())

df = pd.read_csv(os.path.join(examples_path, 'ex1.csv'))  # csv 파일 읽기, 자동으로 첫 행은 column name으로 반영
df

pd.read_table(os.path.join(examples_path, 'ex1.csv'), sep=',')

with open(os.path.join(examples_path, "ex2.csv"), 'r') as f:
  print(f.read())

print(pd.read_csv(os.path.join(examples_path, 'ex2.csv')))  # 자동으로 첫 행은 column name으로 반영
print('\n')
print(pd.read_csv(os.path.join(examples_path, 'ex2.csv'), header=None))  # header 없이 default 설정
print('\n')
print(pd.read_csv(os.path.join(examples_path, 'ex2.csv'), names=['a', 'b', 'c', 'd', 'message']))   # column name 반영

with open(os.path.join(examples_path, "ex3.txt"), 'r') as f:
  print(f.read())
result = pd.read_table(os.path.join(examples_path, 'ex3.txt'), sep='\s+')
result

with open(os.path.join(examples_path, "ex4.csv"), 'r') as f:
  print(f.read())
pd.read_csv(os.path.join(examples_path, 'ex4.csv'), skiprows=[0, 2, 3])  # 특정 row skip

with open(os.path.join(examples_path, "ex5.csv"), 'r') as f:
  print(f.read())
result = pd.read_csv(os.path.join(examples_path, 'ex5.csv'))
print('\n')
print(result)
print('\n')
print(pd.isnull(result))

"""### Reading Text Files in Pieces"""

pd.options.display.max_rows = 10  # pandas 화면 출력 행 개수 설정

result = pd.read_csv(os.path.join(examples_path, 'ex6.csv'))
result

pd.read_csv(os.path.join(examples_path, 'ex6.csv'), nrows=5)  # 처음 N개 행만 읽기

"""### Writing Data to Text Format"""

data = pd.read_csv(os.path.join(examples_path, 'ex5.csv'))
data

data.to_csv(os.path.join(examples_path, 'out.csv'))  # pandas 객체 이용하여 csv 파일 쓰기
with open(os.path.join(examples_path, "out.csv"), 'r') as f:
  print(f.read())

import sys
data.to_csv(sys.stdout, sep='|')  # sys.stdout: 시스템 표준출력

data.to_csv(sys.stdout, index=False, header=False)

data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])

"""## Binary Data Formats"""

frame = pd.read_csv(os.path.join(examples_path, 'ex1.csv'))
print(frame)
frame.to_pickle(os.path.join(examples_path, 'frame_pickle.pkl'))

pd.read_pickle(os.path.join(examples_path, 'frame_pickle.pkl'))

import os
os.remove(os.path.join(examples_path, "frame_pickle.pkl"))

"""## Handling Missing Data"""

string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
print(string_data)
print('\n')
string_data.isnull()

string_data[0] = None
string_data.isnull()

"""### Filtering Out Missing Data"""

from numpy import nan as NA
data = pd.Series([1, NA, 3.5, NA, 7])
print(data)
print('\n')
data.dropna()  # drop rows with nan

data[data.notnull()]  # slicing rows with not null

data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                     [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()  # drop rows with nan
print(data)
print('\n')
print(cleaned)

data.dropna(how='all')  # drop rows with all nans

data[4] = NA
print(data)
print('\n')
data.dropna(axis=1, how='all')  # drops cols with all nans

df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df.iloc[:3, 0] = NA
print(df)
print('\n')
print(df.dropna())
print('\n')
print(df.dropna(thresh=2))  # drops rows with two nans

"""### Filling In Missing Data"""

print(df)
print('\n')
df.fillna(0)  # fill 0 in nans

df.fillna({1: 0.5, 2: 0})  # fill vals via cols

_ = df.fillna(0, inplace=True)  # inplace=True: use the same memory addr
df

data = pd.Series([1., NA, 3.5, NA, 7])
print(data)
data.fillna(data.mean())  # fill mean in nans
print(data)
data.fillna(data.mean(), inplace=True)  # fill mean in nans
print(data)

"""## Data Transformation

### Removing Duplicates
"""

data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                     'k2': [1, 1, 2, 3, 3, 4, 4]})
data

data.duplicated()

data.drop_duplicates()  # drop duplicates(rows) that has the same cols

data['v1'] = range(7)
print(data)
print('\n')
print(data.drop_duplicates(['k1']))  # drop duplicates(rows) using subset
print('\n')
print(data.drop_duplicates(['k2']))  # drop duplicates(rows) using subset

"""### Replacing Values"""

data = pd.Series([1., -999., 2., -999., -1000., 3.])
data

data.replace(-999, np.nan)  # replace A to B

data.replace([-999, -1000], np.nan)

data.replace([-999, -1000], [np.nan, 0])  # element-wise mapping

data.replace({-999: np.nan, -1000: 0})  # key:val mapping

"""### Detecting and Filtering Outliers"""

data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()  # show frequently used statistics

col = data[2]  # slicing subset
col[np.abs(col) > 3]  # slicing absolute(col) > 3

data[(np.abs(data) > 3).any(1)]  # slicing absolute(dataframe) > 3 (at least 1)

data[np.abs(data) > 3] = np.sign(data) * 3  # np.sign: pos/neg
data.describe()

np.sign(data).head()  # show first 5 rows

"""# Data Wrangling: Join, Combine,"""

import numpy as np
import pandas as pd
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

"""## Combining and Merging Datasets

### Database-Style DataFrame Joins
"""

df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
print(df1)
print('\n')
print(df2)

pd.merge(df1, df2)  # merge(inner-join) using the col that has the same name

pd.merge(df1, df2, on='key')  # should indicate the 'on'

df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
                    'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')  # merge(default): inner-join

pd.merge(df1, df2, how='outer')  # merge(outer-join): union set

"""### Concatenating Along an Axis
#### merge: inner-join
#### concat: outer-join
"""

arr = np.arange(12).reshape((3, 4))
print(arr)
print('\n')
np.concatenate([arr, arr], axis=1)

s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])

pd.concat([s1, s2, s3])  # concat via row (outer-join)

pd.concat([s1, s2, s3], axis=1) # concat via col

s4 = pd.concat([s1, s3])
print(s4)
print('\n')
print(pd.concat([s1, s4], axis=1))
print('\n')
print(pd.concat([s1, s4], axis=1, join='inner'))
print('\n')

"""## Reshaping and Pivoting

### Reshaping with Hierarchical Indexing
"""

data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],
                    name='number'))
data

result = data.stack()  # pivot: col -> row
result

result.unstack()  # pivot: row -> col