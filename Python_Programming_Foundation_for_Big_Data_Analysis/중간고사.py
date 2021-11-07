# Data Mining section questions
# 1.
# 지도 학습
# -훈련 데이터로부터 하나의 함수를 유추해내기 위한 기계학습

# 비지도학습
# -훈련 데이터가 주어지지 않고 새로운 데이터가 입력되었을 때 이 데이터가 기존에 분리된 부분 중 어디에 속하는지에 따라 구분하는 군집화에 많이 사용되는 기계학습

# 2.
# 훈련데이터
# -모델 을 학습시키기 위해서 사용되는 데이터

# 검증데이터
# -훈련 데이터를 이용하여 학습된 모델을 평가하고자 사용되는 데이터

# Hold-out method
# -데이터의 약 70%는 훈련 데이터로 사용하고 데이터의 나머지 약 30%는 검증 데이터로 사용하는 방법

# 3.
# 수치 예측
# -예측 하고자 하는 목표 변수를 숫자로 표현한 연속형 데이터
# 예)키

# 범주 예측
# – 예측 하고자 하는 목표 변수를 몇 개의 범주로 구분하는 범주형 데이터
# 예)성별

# Python section questions

4. 5%2 = 1

5. (2**4) // 3 = 5

6.
A = "awdawdawdawd.com"
b = ""
for i in A:
    if i == '@':
        break
    b += str(i)
print(b)

A = "adsadsasdadasd.com"
mail = A[:11]
print(mail)

7.
review = "나는 이 시험 너무 어려워!"
review = review.replace("나는", "강현석은")
review = review.replace("어려워", "쉬워")
print(review)

8.
print([(lambda x, y : '{}'.format(x*y))(x, y) for x in range(2,10) for y in range(1, 10)])

9.
def avg_many():
    s = 0
    n=int(input("입력 숫자의 개수(정수만 입력): "))

    for x in range(1,n+1):
        a= float(input("%d번째 수: "%(x)))
        s+=a

    print("평균: %.3f."%(s/n))

avg_many()

10.
a = [1,2,3,4,5]
b = 0
for i in a:
    b+=1
print('주어진 list의 길이는 %d입니다'%b)