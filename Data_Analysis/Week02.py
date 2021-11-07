# 여러분의 이름과 학번을 출력하세요.
name = "강현석"
id = "20187100"
print(name, id)

# 문제 1. for문과 range 구문을 사용해서 0~9까지 한 라인에 숫자 하나씩 순차적으로 출력하는 프로그램을 작성하라.

# 정답 1.
for i in range(10):
    print(i)

# 문제 2. for문과 range 구문을 사용해서 0~9까지 한 라인에 숫자 하나씩 순차적으로 본인의 이름과 함께 출력하는 프로그램을 작성하라.
# 출력 예시: 0 홍길동

# 정답 2.
for i in range(10):
    print(i,name)

# 문제 3. 월드컵은 4년에 한 번 개최된다. 2002년에 월드컵이 개최되었다.
# range()를 사용하여 2002~2050년까지 중 월드컵이 개최되는 연도를 출력하라.

# 정답 3.
for i in range(2002, 2051, 4):
    print(i)

# 문제 4. 1부터 30까지의 숫자 중 3의 배수를 for문 이용하여 출력하라.

# 정답 4.
for i in range(1,30):
    if i % 3 == 0:
        print(i)

# 문제 5. 1부터 30까지의 숫자 중 3의 배수를 리스트 내포(list comprehension)하여 출력하라.

# 정답 5.
num_list = [x for x in range(3, 31, 3)]
print( num_list)

# 문제 6. 20부터 0까지 1씩 감소하는 숫자들을 리스트 내포하여 출력하라.

# 정답 6.
num_list = [ x for x in range(20, -1, -1)]
print(num_list)

# 문제 7. 구구단 3단을 for문, range() 이용하고 if문 없이 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27

# 정답 7.
for i in range(1, 10, 2):
    print(i* 3)

# 문제 8. 구구단 3단을 for문, range(), if문 이용하여 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27

# 정답 8.
for i in range(1, 10):
    if i % 2 == 1 : 
        print(i * 3)

# 문제 9. 구구단 3단을 리스트 내포(range(step) 포함) 이용하여 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27

# 정답 9.
num_list = [x*3 for x in range(1, 10, 2)]
print(num_list)

# 문제 10. 구구단 3단을 리스트 내포(if문 포함) 이용하여 출력하라. 단 홀수 번째만 출력한다.
# 출력 예시: 3, 9, 15, 21, 27

# 정답 10.
num_list = [x * 3 for x in range(1, 10) if x % 2 == 1]
print(num_list)

# 문제 11. 여러분의 학번과 이름 문자열을 화면에 출력하는 print_student_info() 함수를 정의하라.

# 정답 11.
# 코드 작성: id, name 매개변수 사용하여 표준출력하는 함수
def print_student_info(id, name):
    print(id, name)

# 출력 구문: 수정하지 말 것
print_student_info(id, name)

# 문제 12. 문제 11에서 정의한 print_student_info 함수를 5번 호출하라.

# 정답 12.
for i in range(5):
    print_student_info(id,name)

# 문제 13. 아래의 에러가 발생하는 원인을 찾고 코드를 수정하여 문제를 해결하라.
#hello_hello()
def hello_hello():
  print("Hi")

# 정답 13.
hello_hello()

# 문제 14. 아래 코드의 실행 결과를 예측하여 출력하라.
print("B")

def message():
    print("A")

print("C")
message()

# 정답 14.
# 코드 작성: 아래 print 함수에 예측한 결과를 문자열로 넣기
print("B, C, A")

# 문제 15. 아래 코드의 실행 결과를 예측하여 출력하라.
print("B")
def message1():
    print("A")
print("C")
def message2():
    print("D")
message1()
print("E")
message2()

#정답 15.
# 코드 작성: 아래 print 함수에 예측한 결과를 문자열로 넣기
print("B, C, A, E, D")

# 문제 16. 아래와 같은 에러가 발생하는 원인을 찾고 코드를 수정하여 문제를 해결하라.
# 출력값 예시: 안녕3
# def test(a, b) :
#     print(a + b)

# 정답 16.
def test(a, b) :
    print(a + str(b))
# 출력 구문: 수정하지 말 것
test("에러 싫어요", 3)

# 문제 17. 성적 리스트를 입력받아 평균을 출력하는 print_score 함수를 정의하라.
# sum(): 객체의 숫자 합을 반환하는 내장함수
# len(): 객체의 길이를 반환하는 내장함수

# 정답 17.
# 코드 작성: print_score 함수
def print_score(score_list):
    print(sum(score_list) / len(score_list))
# 출력 구문: 수정하지 말 것
print_score([1, 2, 3])

# 문제 18. 문제 17의 print_score 함수를 키워드 인수 방식(매개변수 지정, [1, 2, 3] 대입)으로 호출하라.

# 정답 18.
print_score(score_list=[1, 2, 3])

# 문제 19. 문제 17의 print_score 함수의 매개변수 초기값을 [1, 1, 1]로 설정하고 인수 없이 호출하라.

# 정답 19.
# 코드 작성: print_score 함수
def print_score(score_list=[1, 1, 1]):
    print(sum(score_list) / len(score_list))

# 출력 구문: 수정하지 말 것
print_score()

# 문제 20. 아래 코드의 실행 결과를 예측하여 출력하라.
a = 3
def vartest(a):
  a = a + 1

vartest(a)
print(a)

# 정답 20.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print(3)

# 문제 21. 아래 코드의 실행 결과를 예측하여 출력하라.
a = [1]
def vartest(a):
    a.append(2)

vartest(a)
print(a)

# 정답 21.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print(1, 2)

# 문제 22. 아래 코드의 실행 결과를 예측하여 출력하라.
a = 1
def vartest():
  global a
  a = a + 1

vartest()
print(a)

# 정답 22.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print(2)

# 문제 23. 아래 코드의 실행 결과를 예측하여 출력하라.
def print_student_info(id, name):
  print(id, name)

id = '20187100'
name = '강현석'
print_student_info(name, id)

# 정답 23.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print('강현석', '20187100')

# 문제 24. 아래 코드의 실행 결과를 예측하여 출력하라.
def print_student_info(id, name):
  print(id, name)

id = '20187100'
name = '강현석'
print_student_info(name=name, id=id)

# 정답 24.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print('20187100', '강현석')

# 문제 25. 아래 코드의 실행 결과를 예측하여 출력하라.
def print_student_info(*args):
  print(args)

id = '20187100'
name = '강현석'
print_student_info(name, id)

# 정답 25.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print(('강현석', '20187100'))

"""## 연습문제

"""

# 문제 26. 1~10까지의 숫자 중 모든 홀수의 합을 출력하는 프로그램을 for 문을 사용하여 작성하라.

# 정답 26.
sum=0
for i in range(1,10,2):
    sum += i
print(sum)

# 문제 27. 두 정수 A와 B가 주어졌을 때, A와 B를 비교하는 프로그램을 작성하시오.
# 입력 시, A에 3, B에 2 입력하세요.
# 문제 링크: https://www.acmicpc.net/problem/1330

# 정답 27.
values = input()
values = values.split()
a = int(values[0])
b = int(values[1])
# 코드 작성: A와 B를 비교하여 결과 출력
if(a > b):  print(">")
elif(a == b):   print("==")
else:   print("<")

# 문제 28. 하나의 문자를 입력받아 문자열 끝에 ":D" 스마일 문자열을 이어 붙여 출력하는
# print_with_smile 함수를 정의하라.
# 출력값 예시: (^ ^):D

# 정답 28.
# 코드 작성: print_with_smile 함수
def print_with_smile (string) :
    print(string + ":D")
# 출력 구문: 수정하지 말 것
print_with_smile("(^ ^)")

# 문제 29. 입력된 문자열을 역순으로 출력하는 print_reverse 함수를 정의하라.

# 정답 29.
# 코드 작성: print_reverse 함수
def print_reverse(string) :
    print(string[::-1])
# 출력 구문: 수정하지 말 것
print_reverse("!!브이바 는오나 서에짬")

# 문제 30. 아래 코드의 오류 원인을 찾고 수정하여 출력값 예시처럼 출력하라.
# 출력값 예시: 홍길동 11112222
def print_student_info(*args):
  a, b = args
  print(a, b)

id = '20187100'
name = '강현석'
print_student_info(name, id)

# 정답 30.

