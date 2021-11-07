# 문제 1. for문과 range 구문을 사용해서 0~9까지 한 라인에 하나씩 순차적으로 출력하는 프로그램을 작성하라.
# 정답 1.
for i in range(10):
  print(i)


# 문제 2. for문과 range 구문을 사용해서 0~9까지 한 라인에 하나씩 순차적으로 본인의 이름과 함께 출력하는 프로그램을 작성하라.
# 출력 예시: 0 홍길동
# 정답 2.
for i in range(10):
  print(i,"강현석")


# 문제 3. 월드컵은 4년에 한 번 개최된다. 2002년에 월드컵이 개최되었다.
# range()를 사용하여 2002~2050년까지 중 월드컵이 개최되는 연도를 출력하라.
# 정답 3.
for i in range(2002, 2051, 4):
  print(i)


# 문제 4. 1부터 30까지의 숫자 중 3의 배수를 for문 이용하여 출력하라.
# 정답 4.
for i in range(1, 11):
  print(i * 3)


# 문제 5. 1부터 30까지의 숫자 중 3의 배수를 리스트 내포(list comprehension)하여 출력하라.
# 정답 5.
values = [i * 3 for i in range(1, 11)]
print(values)


# 문제 6. 20부터 0까지 1씩 감소하는 숫자들을 리스트 내포하여 출력하라.
# 정답 6.
values = [i for i in range(20, -1, -1)]
print(values)
values = []
for i in range(20, -1, -1):
  values.append(i)
print(values)


# 문제 7. 구구단 3단을 for문, range() 이용하고 if문 없이 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27
# 정답 7.
for i in range(1, 10, 2):
  print(i * 3)


# 문제 8. 구구단 3단을 for문, range(), if문 이용하여 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27
# 정답 8.
for i in range(1,10):
  if i % 2 == 1:
    print(i * 3)


# 문제 9. 구구단 3단을 리스트 내포(range(step) 포함) 이용하여 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27
# 정답 9.
values = [i *3 for i in range(1, 10, 2)]
print(values)


# 문제 10. 구구단 3단을 리스트 내포(if문 포함) 이용하여 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27
# 정답 10.
values = [i * 3 for i in range(1, 10) if i % 2 == 1]
print(values)


# 문제 11. 1~10까지의 숫자 중 모든 홀수의 합을 출력하는 프로그램을 for 문을 사용하여 작성하라.
# 정답 11.
values_sum = 0
for i in range (1, 11):
  if i % 2 == 1:
    values_sum +=i
print(values_sum)


# 문제 12. 두 정수 A와 B가 주어졌을 때, A와 B를 비교하는 프로그램을 작성하시오.
# 입력 시, A에 3, B에 2 입력하세요.
# 문제 링크: https://www.acmicpc.net/problem/1330
# 정답 12.
values = input()
values = values.split()
a = int(values[0])
b = int(values[1])
# 코드 작성: A와 B를 비교하여 결과 출력
if a > b:
  print('>')
elif a < b:
  print('<')
else:
  print('==')