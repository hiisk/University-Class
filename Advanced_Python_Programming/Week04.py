# 문제 1. 여러분의 학번과 이름 문자열을 화면에 출력하는 print_student_info() 함수를 정의하라.
# 정답 1.
# 코드 작성: id, name 매개변수 사용하여 표준출력하는 함수
def print_student_info(id,name):
  print(id,name)
  
id = '20187100'  # 수강생분의 학번 입력해주세요
name = '강현석'  # 수강생분의 이름 입력해주세요
print_student_info(id, name)


# 문제 2. 문제 1에서 정의한 print_student_info 함수를 10번 호출하라.
# 정답 2.
for i in range(10):
  print_student_info(id,name)


# 문제 3. 아래의 에러가 발생하는 원인을 찾고 코드를 수정하여 문제를 해결하라.
# hello_hello()
# def hello_hello():
#   print("Hi")
# 정답 3.
def hello_hello():
  print("Hi")

hello_hello()


# 문제 4. 아래 코드의 실행 결과를 예측하여 출력하라.
print("A")
def message():
    print("B")
print("C")
message()
# 정답 4.
# 코드 작성: 아래 print 함수에 예측한 결과를 인수로 넣기
print("A, C, B")


# 문제 5. 아래 코드의 실행 결과를 예측하여 출력하라.
print("A")
def message1() :
    print("B")
print("C")
def message2() :
    print("D")
message1()
print("E")
message2()
#정답 5.
# 코드 작성: 아래 print 함수에 예측한 결과를 인수로 넣기
print("A, C, B, E, D")


# 문제 6. 아래와 같은 에러가 발생하는 원인을 찾고 코드를 수정하여 문제를 해결하라.
# 출력값 예시: 안녕3
def 함수(a, b) :
    print(a + str(b))
함수("안녕", 3)
# 정답 6.


# 문제 7. 하나의 문자를 입력받아 문자열 끝에 ":D" 스마일 문자열을 이어 붙여 출력하는
# print_with_smile 함수를 정의하라.
# 출력값 예시: (^ ^):D
# 정답 7.
# 코드 작성: print_with_smile 함수
def print_with_smile(x):
  print(x + " :D")
print_with_smile("(^ ^)")


# 문제 8. 입력된 문자열을 역순으로 출력하는 print_reverse 함수를 정의하라.
# 정답 8.
# 코드 작성: print_reverse 함수
def print_reverse(sentence):
  print(sentence[::-1])
print_reverse("!!해요필 이썬이파 니으짧 무너 은생인")


# 문제 9. 성적 리스트를 입력받아 평균을 출력하는 print_score 함수를 정의하라.
# sum(): 객체의 숫자 합을 반환하는 내장함수
# len(): 객체의 길이를 반환하는 내장함수
# 정답 9.
# 코드 작성: print_score 함수
def print_score(score_list):
  print(sum(score_list)/len(score_list))
print_score([1, 2, 3])


# 문제 10. 문제 9의 print_score 함수를 키워드 인수 방식(매개변수 지정, [1, 2, 3] 대입)으로 호출하라.
# 정답 10.
print_score(score_list=[1,2,3])


# 문제 11. 문제 9의 print_score 함수의 매개변수 초기값을 [1, 1, 1]로 설정하고 인수 없이 호출하라.
# 정답 11.
# 코드 작성: print_score 함수
def print_score(score_list=[1,1,1]):
  print(sum(score_list)/len(score_list))
print_score()


# 문제 12. 아래 코드의 실행 결과를 예측하여 출력하라.
a = 1
def vartest(a):
  a = a + 1
vartest(a)
print(a)
# 정답 12.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print("1")


# 문제 13. 아래 코드의 실행 결과를 예측하여 출력하라.
a = [1]
def vartest(a):
  a.append(2)
vartest(a)
print(a)
# 정답 13.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print([1,2])


# 문제 14. 아래 코드의 실행 결과를 예측하여 출력하라.
a = 1
def vartest():
  global a
  a = a + 1
vartest()
print(a)
# 정답 14.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print(2)


# 문제 15. 아래 코드의 실행 결과를 예측하여 출력하라.
def print_student_info(id, name):
  print(id, name)
id = '20200000'
name = '홍길동'
print_student_info(name, id)
# 정답 15.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print('홍길동 20200000')


# 문제 16. 아래 코드의 실행 결과를 예측하여 출력하라.
def print_student_info(id, name):
  print(id, name)
id = '20200000'
name = '홍길동'
print_student_info(name=name, id=id)
# 정답 16.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print('20200000 홍길동')


# 문제 17. 아래 코드의 실행 결과를 예측하여 출력하라.
def print_student_info(*args):
  print(args)
id = '20200000'
name = '홍길동'
print_student_info(name, id)
# 정답 17.
# 코드 작성: print 함수의 인수에 예측 결과를 반영
print(('홍길동', '20200000'))


# 문제 18. 아래 코드의 오류 원인을 찾고 수정하여 출력값 예시처럼 출력하라.
# 출력값 예시: 홍길동 20200000
# def print_student_info(*args):
#   a, b, c = args
#   print(a, b)
# id = '20200000'
# name = '홍길동'
# print_student_info(name, id)
# 정답 18.
def print_student_info(*args):
  a, b = args
  print(a, b)
id = '20200000'
name = '홍길동'
print_student_info(name, id)


# 문제 19. 아래 코드의 실행 결과를 예측하여 출력하라.
def print_student_info(name='임꺽정', *args, **kwargs):
  print(name, args, kwargs)
id = '20200000'
name = '홍길동'
city = '대전'
math_score = '100'
science_score = '90'
print_student_info(name, id, city, math_score=math_score,
                   science_score=science_score)
# 정답 19.
# 코드 작성: 아래 print 함수의 인수에 예측 결과를 반영
print('홍길동',('20200000','대전'), {'math_score': '100', 'science_score':'90'})