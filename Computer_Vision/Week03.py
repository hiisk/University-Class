print("강현석", "20187100")

"""# Getting Started with pandas, numpy, and matplotlib"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

"""# Pandas 실습

### Series
"""

obj = pd.Series([4, 7, -5, 3])
obj

print(obj.values)
print(obj.index)  # like range(4)

obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])  # index 임의로 설정하기
obj2

obj2.index

print(obj2['a'])
obj2['d'] = 6
print(obj2[['c', 'a', 'd']])

print(obj2 > 0)
obj2[obj2 > 0]

print(obj2 * 2)
print(np.exp(obj2))  # exponential 함수

print('b' in obj2.index)
print('e' in obj2.index)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4

pd.isnull(obj4)  # null 있는지?

pd.notnull(obj4)  # null 없는지?

obj4.isnull()  # null 있는지?

"""### DataFrame"""

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)

frame

frame.head()  # 데이터프레임 샘플 확인

frame.tail()  # 데이터프레임 샘플 확인

frame.describe()  # 수치형 데이터에 대해 주요 통계치 출력

frame = frame.astype({'year': str})  # 수치형 데이터를 문자열 데이터로 타입 변경 (one-hot vector 생성을 위함)
frame

pd.get_dummies(frame)  # 카테고리 문자열 데이터를 one-hot vector로 변환

pd.DataFrame(data, columns=['year', 'state', 'pop'])

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
print(frame2)
print(frame2.columns)
print(frame2.index)
# 리스트(list): 새로운 라벨 인덱싱하면 오류 발생
# 딕셔너리(dictionary): 새로운 라벨 인덱싱하면 새로운 라벨 생성

frame2['state']  # 데이터프레임의 'state' 열의 정보(index, values) 출력

frame2.year  # 데이터프레임의 'year' 열의 정보 출력

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

frame3.T  # 데이터프레임 transpose 시키기

pd.DataFrame(pop, index=[2001, 2002, 2003])

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)

"""# NumPy 실습"""

my_arr = np.arange(1000000)
my_list = list(range(1000000))

# Commented out IPython magic to ensure Python compatibility.
# %time for _ in range(10): my_arr2 = my_arr * 2  # ndarray 연산 시간 측정
# %time for _ in range(10): my_list2 = [x * 2 for x in my_list]  # list 연산 시간 측정

"""## The NumPy ndarray: A Multidimensional Array Object"""

import numpy as np

# 문제 1. 표준 정규 분포(평균 0, 표준편차 1)에서 난수 ndarray 생성하기
# ndarray 이름: data
# ndarray 크기: (2, 3)

# 정답 1.
data = np.random.randn(2,3)
print(data)

# 문제 2. 0부터 1사이의 균일 분포에서 난수 ndarray 생성하기
# ndarray 이름: data2
# ndarray 크기: (2, 3)

# 정답 2.
data2 = np.random.randn(2,3)
print(data2)

# 문제 3. 균일 분포에서 정수 난수 1개 생성하기
# 난수 객체 이름: data3
# 난수 범위: (1, 20)

# 정답 3.
data3 = np.random.randint(1, 20)
print(data3)

# 문제 4. data 배열에 10 곱하기

# 정답 4.
data * 10

# 문제 5. data 배열에 data 배열 더하기

# 정답 5.
data + data

# 문제 6. data 배열 크기 출력하기

# 정답 6.
data.shape

# 문제 7. data 데이터타입 출력하기

# 정답 7.
type(data)

"""### Creating ndarrays"""

# 문제 8. 아래 리스트를 이용하여 NumPy ndarray 생성하기
# ndarray 이름: arr1
data1 = [6, 7.5, 8, 0, 1]

# 정답 8.
arr1 = np.array(data1)
print(arr1)

# 문제 9. 아래 리스트를 이용하여 ndarray 생성하기
# ndarray 이름: arr2
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]

# 정답 9.
arr2 = np.array(data2)
print(arr2)

# 문제 10. 크기 10이고 0으로 초기화한 ndarray 생성하기
# ndarray 이름: data_zero

# 정답 10.
data_zero = np.zeros(10)
print(data_zero)

# 문제 11. 크기 (3, 6)이고 0으로 초기화한 ndarray 생성하기
# ndarray 이름: data_zero1

# 정답 11.
data_zero1 = np.zeros((3, 6))
print(data_zero1)

# 문제 12. 크기 (2, 3, 2)이고 초기화하지 않은 ndarray 생성하기
# ndarray 이름: data_empty

# 정답 12.
data_empty = np.empty((2, 3, 2))
print(data_empty)

# 문제 13. 0 부터 15 전까지 시퀀스 배열 생성하기

# 정답 13.

np.arange(15)

# 문제 14. 0 부터 2 전까지 스텝크기 0.1 인 시퀀스 배열 생성하기

# 정답 14.
np.arange(0, 2, 0.1)

"""### Data Types for ndarrays"""

# 문제 15. 아래의 리스트를 이용하여 데이터타입이 np.float64인 ndarray 생성하기
# ndarray 이름: arr1
data_list = [1, 2, 3]

# 정답 15.
arr1 = np.array(data_list, dtype=np.float64)
print(arr1, arr1.dtype)

# 문제 16. 아래의 리스트를 이용하여 데이터타입이 np.int32인 ndarray 생성하기
# ndarray 이름: arr2
data_list = [1, 2, 3, 4, 5]

# 정답 16.
arr2 = np.array(data_list, dtype=np.int32)
print(arr2, arr2.dtype)

# 문제 17. 문제 16에서 생성한 arr2의 데이터타입을 np.float64 로 변환한 ndarray의 데이터타입 출력하기
# ndarray 이름: float_arr

# 정답 17.
float_arr = np.array(arr2,dtype=np.float64 )
print(float_arr, float_arr.dtype)

# 문제 18. 아래의 ndarray를 np.int32 로 변환한 ndarray 생성하기
# ndarray 이름: int_arr
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])

# 정답 18.
int_arr = np.array(arr, dtype=np.int32 )
print(int_arr, int_arr.dtype)

# 문제 19. 아래의 ndarray를 np.float64로 변환한 ndarray 생성하기
# ndarray 이름: float_arr
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

# 정답 19.
float_arr = numeric_strings.astype(np.float64) 
print(float_arr, float_arr.dtype)

"""### Arithmetic with NumPy Arrays"""

# 문제 20. 아래 2개의 ndarray 간의 element-wise 곱하기
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[4., 5., 6.], [1., 2., 3.]])

# 정답 20.
arr1* arr2

# 문제 21. 아래 2개의 ndarray 간의 element-wise 나누기
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[4., 5., 6.], [1., 2., 3.]])

# 정답 21.
arr1 / arr2

# 문제 22. 아래 ndarray에 element-wise 0.5 거듭제곱하기
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])

# 정답 22.
arr1 ** 0.5

# 문제 23. 아래 ndarray을 transpose하기
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])

# 정답 23.
np.transpose(arr1)

# 문제 24. 아래 2개의 ndarray을 행렬 곱하여 arr3 생성하기
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = arr1.T

# 정답 24.
arr3 = np.matmul(arr1, arr2)
print(arr3)

# 문제 25. arr1이 3.5보다 큰지에 대해 불린 배열 출력하기
arr1 = np.array([1., 2., 3., 4., 5., 6.])

# 정답 25.
arr1 > 3.5

# 문제 26. arr1이 3.5보다 큰 배열 인덱스 출력하기
arr1 = np.array([1., 2., 3., 4., 5., 6.])

# 정답 26.
np.where(arr1 > 3.5)

# 문제 27. arr2중에서 가장 큰 값 출력하기
arr2 = np.array([4., 5., 6., 1., 2., 3.])

# 정답 27.
np.amax(arr2)

# 문제 28. arr2중에서 가장 큰 값 인덱스 출력하기
arr2 = np.array([4., 5., 6., 1., 2., 3.])

# 정답 28.
np.argmax(arr2)

# 문제 29. arr2 오름차순으로 정렬하기
arr2 = np.array([4., 5., 6., 1., 2., 3.])

# 정답 29.
np.sort(arr2)

# 문제 30. arr2 오름차순으로 정렬했을 때의 인덱스 출력하기
arr2 = np.array([4., 5., 6., 1., 2., 3.])

# 정답 30.
np.argsort(arr2)

"""### Basic Indexing and Slicing"""

# 문제 31. 아래 arr 배열 인덱스 5부터 8전까지의 값을 12로 변경하기 
arr = np.arange(10)

# 정답 31.
arr[5:8] = 12
print(arr)

# 문제 32. 문제 29의 arr 배열의 인덱스 5부터 8전까지를 슬라이싱하여 arr_slice 배열 생성하기

# 정답 32.
arr_slice = arr[5:8]
print(arr_slice)

# 문제 33. 문제 30의 arr_slice의 인덱스 1번째 값을 12345로 변경하고 arr 배열 출력하기
# arr 배열의 값이 왜 변경되었을까 생각해보기

# 정답 33.
arr_slice[1] = 12345
print(arr)

# 문제 34. 문제 31의 arr_slice 배열의 모든 값을 64로 변경하고 arr 배열 출력하기

# 정답 34.
arr_slice[:] = 64
print(arr)

# 문제 35. 문제 31의 arr_slice 배열의 복사본을 생성하고 모든 값을 9999로 변경하고 arr 배열 출력하기

# 정답 35.
arr_slice = arr_slice.copy()
arr_slice[:] = 9999
print(arr)
print(arr_slice)

