# colab
#1-1


#구글 드라이브와 연동
from google.colab import drive
drive.mount('/gdrive', force_remount=True)


#구글 드라이브 폴더 확인
# !ls "/gdrive/My Drive"


#랜덤
import numpy as np
np.random.seed(20187100)
A = np.random.rand(5,5)
print(A)
print("20187100_강현석")
np.savetxt("/gdrive/My Drive/Colab Notebooks/data/save.csv", A, fmt= '%f', delimiter=',')
B = np.loadtxt("/gdrive/My Drive/Colab Notebooks/data/save.csv",delimiter=',')
print(B)


#1-2


my_list = [10, 'hello list', 20]
print(my_list[1])
#hello list


my_list_2 =[[10, 20, 30], [40, 50, 60]]
print(my_list_2[1][2])
#60


import numpy as np
print(np.__version__)
#1.18.5


my_arr = np.array([[10, 20, 30], [40, 50, 60]])
print(my_arr)
#[[10 20 30]
#[40 50 60]]


type(my_arr)
#numpy.ndarray


print(my_arr[0][2])
#30


np.sum(my_arr)
import matplotlib.pyplot as plt
values = np.array([[1, 2, 3, 4, 5],[1, 4, 9, 16, 25]])
plt.plot(values[0, :],values[1, :])
plt.show()


np.random.seed(20187100)
random_values = np.random.randn(20,20)
plt.scatter(random_values[0, :], random_values[1, :])
print("20187100_강현석")


#구글 드라이브와 연동
from google.colab import drive
drive.mount("/gdrive", force_remount=True)


#csv파일 불러오기
import pandas as pd
df = pd.read_csv("/gdrive/My Drive/Colab Notebooks/data/gapminder.csv", sep='\t')


print(df.head())
# country continent year lifeExp pop gdpPercap
# 0 Afghanistan Asia 1952 28.801 8425333 779.445314
# 1 Afghanistan Asia 1957 30.332 9240934 820.853030


print(type(df))
#<class 'pandas.core.frame.DataFrame'>


print(df.shape)
#(1704, 6)


print(df.columns)
#Index(['country', 'continent', 'year', 'lifeExp', 'pop', 'gdpPercap'], dtype='object')


print(df.dtypes)
# country object
# continent object
# year int64
# lifeExp float64
# pop int64
# gdpPercap float64
# dtype: object


print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1704 entries, 0 to 1703
# Data columns (total 6 columns):
# # Column Non-Null Count Dtype
# --- ------ -------------- -----
# 0 country 1704 non-null object
# 1 continent 1704 non-null object
# 2 year 1704 non-null int64
# 3 lifeExp 1704 non-null float64
# 4 pop 1704 non-null int64
# 5 gdpPercap 1704 non-null float64
# dtypes: float64(2), int64(2), object(2)
# memory usage: 80.0+ KB
# None


country_df =df['country']
print(country_df.head())
# 0 Afghanistan
# 1 Afghanistan
# 2 Afghanistan
# 3 Afghanistan
# 4 Afghanistan
# Name: country, dtype: object


print(country_df.tail())
# 1699 Zimbabwe
# 1700 Zimbabwe
# 1701 Zimbabwe
# 1702 Zimbabwe
# 1703 Zimbabwe
# Name: country, dtype: object


subset = df[['country', 'continent', 'year']]
print(subset.head())
# country continent year
# 0 Afghanistan Asia 1952
# 1 Afghanistan Asia 1957
# 2 Afghanistan Asia 1962
# 3 Afghanistan Asia 1967
# 4 Afghanistan Asia 1972


print(subset.tail())
# country continent year
# 1699 Zimbabwe Africa 1987
# 1700 Zimbabwe Africa 1992
# 1701 Zimbabwe Africa 1997
# 1702 Zimbabwe Africa 2002
# 1703 Zimbabwe Africa 2007


number_of_rows = df.shape[0]
last_row_index = number_of_rows - 1
print(df.loc[last_row_index])
# country Zimbabwe
# continent Africa
# year 2007
# lifeExp 43.487
# pop 12311143
# gdpPercap 469.709
# Name: 1703, dtype: object


print(df.head(n=1))
# country continent year lifeExp pop gdpPercap
# Afghanistan Asia 1952 28.801 8425333 779.445314


print(df.tail(n=2))
# country continent year lifeExp pop gdpPercap
# 1702 Zimbabwe Africa 2002 39.989 11926563 672.038623
# 1703 Zimbabwe Africa 2007 43.487 12311143 469.709298


print(df.loc[[0, 99, 999]])
# country continent year lifeExp pop gdpPercap
# 0 Afghanistan Asia 1952 28.801 8425333 779.445314
# 99 Bangladesh Asia 1967 43.453 62821884 721.186086
# 999 Mongolia Asia 1967 51.253 1149500 1226.041130


print(df.iloc[1])
# country Afghanistan
# continent Asia
# year 1957
# lifeExp 30.332
# pop 9240934
# gdpPercap 820.853
# Name: 1, dtype: object


print(df.iloc[99])
# country Bangladesh
# continent Asia
# year 1967
# lifeExp 43.453
# pop 62821884
# gdpPercap 721.186
# Name: 99, dtype: object

# print(df.iloc[[0, 99, 999]])
# country continent year lifeExp pop gdpPercap
# 0 Afghanistan Asia 1952 28.801 8425333 779.445314
# 99 Bangladesh Asia 1967 43.453 62821884 721.186086
# 999 Mongolia Asia 1967 51.253 1149500 1226.041130


subset = df.loc[:, ['year', 'pop']]
print(subset.head())
# year pop
# 0 1952 8425333
# 1 1957 9240934
# 2 1962 10267083
# 3 1967 11537966
# 4 1972 13079460


subset = df.iloc[:,[2, 4, -1]]
print(subset.head())
# year pop gdpPercap
# 0 1952 8425333 779.445314
# 1 1957 9240934 820.853030