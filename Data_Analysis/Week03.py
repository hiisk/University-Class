# 여러분의 학번과 이름 문자열을 화면에 출력해주세요.

id = '20187100'  # 수강생분의 학번 입력해주세요
name = '강현석'  # 수강생분의 이름 입력해주세요
print(id, name)

# 구글 드라이브 사용 권한 설정
from google.colab import drive
drive.mount('/gdrive')

# 과제 파일 저장한 폴더 위치 설정 (폴더 위치에 맞춰서 변경할 것)
import os
colab_path = "/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/DA_과제3"  # 경로예시: 구글드라이브/내 드라이브/Colab Notebooks/과제폴더명

# 문제 1. 사람 (Human) 클래스에 "응애응애"를 출력하는 생성자를 추가하라.

# 출력 예제
# >> baby = Human()
# 응애응애

# 정답 1.
# 코드 작성: Human 클래스에 생성자 추가하기
class Human():
    def __init__(self):
        print("응애응애")

# 실행 구문 (아래 코드를 수정하지 마시오.)
baby = Human()

# 문제 2. 사람 (Human) 클래스에 (이름, 나이, 성별)을 받아서 인스턴스 변수(name, age, sex)를
# 초기화하는 생성자를 추가하라.

# 출력 예제
# >> student = Human("홍길동", "20", "남자")
# >> print(student.name, student.age, student.sex)
# 홍길동 20 남자

# 정답 2.
# 코드 작성: Human 클래스에 생성자 추가하기
class Human():
    def __init__(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex

# 실행 구문 (아래 코드를 수정하지 마시오.)
student = Human("홍길동", "20", "남자")
print(student.name, student.age, student.sex)

# 문제 3. 사람 (Human) 클래스에서 이름, 나이, 성별을 출력하는 who() 메소드를 추가하세요.

# 출력 예제
# >> student = Human("홍길동", "20", "남자")
# >> student.who()
# 이름: 홍길동, 나이: 20, 성별: 남자

# 정답 3.
# 코드 작성: Human 클래스에 who() 메소드 추가
class Human():
    def __init__(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex

    def who(self):
        # print(f'이름: {self.name}, 나이: {self.age}, 성별: {self.sex} ')
        # print('이름: {}, 나이: {}, 성별: {}'.format(self.name, self.age, self.sex))
        print('이름: %s, 나이: %s, 성별: %s'%(self.name, self.age, self.sex))
                                                    


# 실행 구문 (아래 코드를 수정하지 마시오.)
student = Human("홍길동", "20", "남자")
student.who()

# 문제 4. 사람 (Human) 클래스에서 (이름, 나이, 성별)을 받는 set_info() 메소드를 추가하세요.

# 출력 예제
# >> student = Human()
# >> student.set_info("홍길동", 20, "남자")
# >> student.who()
# 이름: 홍길동, 나이: 20, 성별: 남자

# 정답 4.
# 코드 작성: Human 클래스에 set_info() 메소드 추가
class Human():
    def who(self):
        print(f"이름: {self.name}, 나이: {self.age}, 성별: {self.sex}")

    def set_info(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex

# 실행 구문 (아래 코드를 수정하지 마시오.)
student = Human()
student.set_info("홍길동", "20", "남자")
student.who()

# 문제 5. 아래 코드의 에러의 원인을 찾고 해결하시오.
# class OMG:
#     def print():
#         print("Oh my god")

# 정답 5.
class OMG:
    def print(self):
        print("Oh my god")
# 실행 구문 (아래 코드를 수정하지 마시오.)
mystock = OMG()
mystock.print()

# 문제 6. 아래의 Car 클래스로 생성한 2개 객체의 클래스 변수 값을 확인하라.
class Car:
    brand = ""

    def __init__(self, num_wheels, price):
        self.num_wheels = num_wheels
        self.price = price

    def get_info(self):
        print(f"바퀴수: {self.num_wheels}, 가격: {self.price}")

sample_1 = Car(2, 1000)
sample_2 = Car(4, 2000)
Car.brand = "삼천리"  # 클래스 변수 값 변경

print(sample_1.brand, sample_2.brand)  # 클래스 변수는 모든 객체에 공유
print(f"두 객체의 클래스 변수의 메모리 주소는 동일한가? {sample_1.brand is sample_2.brand}")

# 문제 7. 난수 생성 모듈 random을 불러오고 0~100 사이의 정수 난수 1개를 생성하라.

# 정답 7.
import random
print(random.randint(0, 100))

# 문제 8. 시간 모듈 time을 불러오고 현재 시각을 출력하라.

# 정답 8.
import time
print(time.localtime())

# 문제 9. 웹 주소 정보 모듈 urllib을 불러오고 "https://www.naver.com" 정보를 불러오시오.

# 정답 9.
import urllib
response = urllib.request.urlopen("https://www.naver.com")
print(response.read())

# 문제 10. readline 함수 이용하여 "newfile.txt" 파일 읽어서 첫 번째 줄 출력하시오.
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "newfile.txt") => "colab_path/newfile.txt"

# 정답 10.
f = open(os.path.join(colab_path, "newfile.txt"), 'r')
line = f.readline()
print(line)
f.close()

# 문제 11. readline 함수 이용하여 "newfile.txt" 파일 읽어서 모든 줄 화면 출력하시오.
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "newfile.txt")

# 정답 11.
f = open(os.path.join(colab_path, "newfile.txt"), 'r')
while True:
    line = f.readline()
    if not line:
        break
    else:
        print(line)
f.close()

# 문제 12. write 함수 이용하여 "foo.txt" 파일에 "Life is too short, you need python" 출력하시오.
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "foo.txt")

# 정답 12.
f = open(os.path.join(colab_path, "foo.txt"), 'w')
f.write("Life is too short, you need python")
f.close()

# 실행 구문 (아래 코드를 수정하지 마시오.)
os.path.exists(os.path.join(colab_path, "foo.txt"))

# 문제 13. with 문 이용하여 문제 12 수행하시오.
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "foo.txt")

# 정답 13.
with open(os.path.join(colab_path, "foo.txt"), 'w') as f:
    f.write("Life is too short, you need python")
    
# 실행 구문 (아래 코드를 수정하지 마시오.)
os.path.exists(os.path.join(colab_path, "foo.txt"))

# 문제 14. pickle 모듈 불러와서 test 객체를 바이너리 파일 출력(wb)하시오.
# 출력할 파일명: "list.pickle"
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "list.pickle")
test = [1, 2, 3, 4, 5]

# 정답 14.
import pickle
with open(os.path.join(colab_path, "list.pickle"), 'wb') as f:
    pickle.dump(test, f)

# 실행 구문 (아래 코드를 수정하지 마시오.)
os.path.exists(os.path.join(colab_path, "list.pickle"))

# 문제 15. 아래 문자열 값을 실수로 변환할 때 에러가 발생한다. 예외처리를 통해 에러가 발생하는 문자열은 0으로 출력하시오.
sample_list = ["3.14", "", "10.27"]

# 에러 발생 영역
# for i in sample_list:
#     print(float(i))

# 정답 15.
for i in sample_list:
    try:
        print(float(i))
    except:
        print(0)

# 문제 16. 어떤 값을 0으로 나누면 ZeroDivisionError 에러가 발생한다.
# try~except로 모든 에러에 대해 예외처리하지 말고
# ZeroDivisionError 에러만 예외처리하라.
# 예외 발생 경고 메시지: "0으로 나누지 마시오."

# 에러 발생 영역
# b = 4 / 0

# 정답 16.
try:
    b = 4 / 0
except ZeroDivisionError:
    print("0으로 나누지 마시오.")

# 문제 17. 어떤 값을 0으로 나누면 ZeroDivisionError 에러가 발생한다.
# try~except로 모든 에러에 대해 예외처리하지 말고
# ZeroDivisionError 에러만 예외처리하여 오류 메시지를 출력하라.

# 에러 발생 영역
# b = 4 / 0

# 정답 17.
try:
    b = 4 / 0
except ZeroDivisionError as e:
    print(e)

# 문제 18. raise 명령어 사용하여
# Eagle 클래스의 fly() 메소드 실행 시
# NotImplementedError 오류 강제로 발생시켜라.
# 출력예시: "fly 메소드를 구현하지 않았습니다."

# 정답 18.
class Eagle():
    def fly(self):
        # 코드 작성 부분
        raise NotImplementedError("fly 메소드를 구현하지 않았습니다")


# 실행 구문(아래 코드를 수정하지 마시오.)
eagle = Eagle()
eagle.fly()

# 문제 19. assert 명령어 사용하여 0으로 나누기 전에 오류 발생시켜라.
# class Calculator():
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#     def div(self):
#         return self.a / self.b

# 정답 19.
class Calculator():
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def div(self):
        assert (self.b != 0)
        return self.a / self.b

# 실행 구문(아래 코드를 수정하지 마시오.)
calc = Calculator(10, 0)
calc.div()

# 문제 20. 아래 리스트의
# 1) 모두 참이면 True 하나라도 거짓이면 False 출력하라.
# 2) 모두 거짓이면 False 출력하라.
sample = [-3, -1.2, 0, 4, -0.5]

# 정답 20.
print(all(sample))
print(any(sample))

# 문제 21. 하나의 함수를 사용하여 아래 두 개의 숫자를 나눈 몫과 나머지를 출력하라.
a = 7
b = 3

# 정답 21.
print(divmod(a, b))

# 문제 22. 아래의 리스트를 순서와 값을 한번에 출력하는 반복문을 작성하라.
sample = ['body', 'foo', 'bar']

# 정답 22.
for i, name in enumerate(sample):
    print(i, name)

# 문제 23. 다음의 함수에 입력되었을 때 반환 값이 참인 것만 묶어서 출력하라.
# filter 함수 사용할 것
def positive(x):
    return x > 0

sample = [-3, -1.2, 0, 4, -0.5]

# 정답 23.
print(list(filter(positive, sample)))

# 문제 24. 다음 리스트 길이를 출력하라.
A = range(10)
B = range(10, 30, 2)

# 정답 24.
print(len(A))
print(len(B))

# 문제 25. 아래의 인스턴스 a, b가 Person 클래스의 인스턴스인지 확인하라.
class Person:
    pass
class Animal:
    pass

a = Person()
b = Animal()

# 정답 25.
print(isinstance(a, Person))
print(isinstance(b, Person))

# 문제 26. 아래의 리스트를 오름차순으로 정렬하여 출력하라.
sample_1 = [3, 1, 2]
sample_2 = ['a', 'c', 'b']
sample_3 = "zeros"
sample_4 = (3, 2, 1)

# 정답 26.
print(sorted(sample_1))
print(sorted(sample_2))
print(sorted(sample_3))
print(sorted(sample_4))

# 문제 27. 아래의 값들의 자료형을 출력하라.
sample_1 = [3, 1, 2]
sample_2 = 'a'
sample_3 = "zeros"
sample_4 = (3, 2, 1)
sample_5 = {'a': 1, 'b': 2}

# 정답 27.
print(type(sample_1))
print(type(sample_2))
print(type(sample_3))
print(type(sample_4))
print(type(sample_5))

"""# 실습 과제"""

# 문제 28. 다음 코드가 동작하도록 Car 클래스를 정의하라.
# 출력 예제
# >> sample = Car(2, 1000)
# >> print(sample.num_wheels, sample.price)
# 2 1000

# 정답 28.
# 코드 작성: Car 클래스
class Car:
    def __init__(self, wheels, price):
        self.num_wheels = wheels
        self.price = price

# 실행 구문 (아래 코드를 수정하지 마시오.)
sample = Car(2, 1000)
print(sample.num_wheels, sample.price)

# 문제 29. 문제 28의 Car 클래스를 상속받은 Bicycle 클래스를 정의하라.
# 출력 예제
# >> sample = Bicycle(2, 1000)
# >> print(sample.num_wheels, sample.price)
# 2 1000

# 정답 29.
# 코드 작성: Car 클래스 상속받은 Bicycle 클래스
class Car:
    def __init__(self, wheels, price):
        self.num_wheels = wheels
        self.price = price


class Bicycle(Car):
    def __init__(self, wheels, price):
        self.num_wheels = wheels
        self.price = price
        
# 실행 구문 (아래 코드를 수정하지 마시오.)
sample = Bicycle(2, 1000)
print(sample.num_wheels, sample.price)

# 문제 30. 다음의 코드가 동작하도록 Bicycle 클래스를 정의하라.
# Bicycle 클래스는 앞의 Car 클래스를 상속받는다.
# 출력 예제
# >> sample = Bicycle(2, 1000, "삼천리")
# >> sample.info()
# 바퀴수: 2, 가격: 1000, 브랜드: 삼천리

# 정답 30.
# 코드 작성: Car 클래스 상속받은 Bicycle 클래스
class Car:
    def __init__(self, wheels, price):
        self.num_wheels = wheels
        self.price = price

class Bicycle(Car):
    def __init__(self, wheels, price, brand):
        self.num_wheels = wheels
        self.price = price
        self.brand = brand

    def info(self):
        print(f"바퀴수: {self.num_wheels}, 가격: {self.price}, 브랜드: {self.brand}")

# 실행 구문 (아래 코드를 수정하지 마시오.)
sample = Bicycle(2, 1000, "삼천리")
sample.info()

# 문제 31. 다음의 Car 클래스를 상속받고 아래의 출력예제가 동작하도록 Bicycle 클래스를 정의하라.
class Car:
    def __init__(self):
        self.num_wheels = 0
        self.price = 0

    def set_info(self, num_wheels, price):
        self.num_wheels = num_wheels
        self.price = price

    def get_info(self):
        print(f"바퀴수: {self.num_wheels}, 가격: {self.price}")

# 출력 예제
# >> sample = Bicycle()
# >> sample.set_info(2, 1000, "삼천리")
# >> sample.get_info()
# 바퀴수: 2, 가격: 1000, 브랜드: 삼천리

# 정답 31.
# 코드 작성: Car 클래스를 상속받는 Bicycle 클래스
class Bicycle(Car):
    def __init__(self):
        self.num_wheels = 0
        self.price = 0
        self.brand = 0
     
    def set_info(self, num_wheels, price, brand):
        self.num_wheels = num_wheels
        self.price = price
        self.brand = brand

    def get_info(self):
        print(f"바퀴수: {self.num_wheels}, 가격: {self.price}, 브랜드: {self.brand}")


# 실행 구문 (아래 코드를 수정하지 마시오.)
sample = Bicycle()
sample.set_info(2, 1000, "삼천리")
sample.get_info()

# 문제 32. 국어, 영어, 수학 점수를 사용자 입력받아 합계를 구하는 객체지향 코드(클래스 사용)를 작성하라.
# 이 때, 학생 클래스의 객체는 객체 생성 시 국어, 영어, 수학 점수를 저장하며, 총점을 구하는 메서드를 작성한다.
# 입력 예제: 입력값으로 국어는 80점, 영어는 70점, 수학은 90점을 사용한다.
# 출력 예제: 세 과목 총점: 240

# 정답 32.
# 코드 작성: Student 클래스를 구현하라.
class Student:
    def __init__(self, kor, math, eng):
        self.kor = kor
        self.math = math
        self.eng = eng
 
    def kor(self):
        return self.kor
 
    def mat(self):
        return self.math
 
    def eng(self):
        return self.eng

    def set_sum(self):
        self.sum = self.kor + self.math + self.eng
        return self.sum

    def print_sum(self):
        print(self.sum)

# 실행 구문 (아래 코드를 수정하지 마시오.)
print("국어, 영어, 수학 점수를 차례로 입력하시오.\n")
score = input()
score = score.split()
score = list(map(int, score))
student_1 = Student(score[0], score[1], score[2])
student_1.set_sum()
student_1.print_sum()

1 # 문제 33. 난수 생성 모듈 random을 불러오고 0과 1 사이의 난수를 생성하라.

# 정답 33.
import random
print(random.random())

# 문제 34. read 함수 이용하여 "newfile.txt" 파일 읽어서 모든 줄 화면 출력하시오.
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "newfile.txt")

# 정답 34.
f = open(os.path.join(colab_path, "newfile.txt"), 'r')
while True:
    line = f.readline()
    if not line:
        break
    else:
        print(line)
f.close()

# 문제 35. os 모듈 불러와서 "log" 디렉토리(폴더) 생성하시오.
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "log")
import os

if os.path.isdir(os.path.join(colab_path, "log")):
    os.rmdir(os.path.join(colab_path, "log"))

# 정답 35.
if not os.path.isdir(os.path.join(colab_path, "log")):
    os.mkdir(os.path.join(colab_path, "log"))
    
# 실행 구문(아래 코드를 수정하지 마시오.)
os.path.exists(os.path.join(colab_path, "log"))

# 문제 36. pickle 모듈 불러와서 생성한 파일을 읽어서 test 객체를 생성하시오.
# 읽어올 파일명: "list.pickle"
# 파일 경로 설정 코드 예제: os.path.join(colab_path, "list.pickle")

# 정답 36.
import pickle

f = open(os.path.join(colab_path, "list.pickle"), "wb")
test = [1, 2, 3, 4, 5]
pickle.dump(test,f)
f.close
# 실행 구문(아래 코드를 수정하지 마시오.)
print(test)

# 문제 37. 아래 try~except 문에 finally 문을 추가하라.
# finally 문: 예외 발생 여부 상관없이 수행하는 구문
# finally 문에서 출력할 메시지: "div 함수 종료합니다.")

# 정답 37.
def div(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError as e:
        print(e)
    # 코드 작성 영역
    finally:
        print("div 함수 종료합니다.")

div(10, 0)

# 문제 38. 어떤 값을 0으로 나누면 ZeroDivisionError 에러가 발생한다.
# ZeroDivisionError 에러 발생하면 그냥 통과시켜라. (오류 회피하기)

# 에러 발생 영역
# b = 4 / 0

# 정답 38.

try:
    b = 4 / 0
except ZeroDivisionError:
    pass

# 문제 39. 아래의 문자열 수식을 실행하여 결과값을 출력하라.
sample_1 = '1+2'
sample_2 = 'divmod(4, 3)'

# 정답 39.
print(eval(sample_1))
print(eval(sample_2))

# 문제 40. 다음 리스트에서 값이 양수인지(True or False) 출력하라.
# map, lambda 함수 사용할 것
sample = [-3, -1.2, 0, 4, -0.5]

# 정답 40.
list(map(lambda a : a > 0, sample))

# 문제 41. 다음 리스트에서 값이 양수인 것만 묶어서 출력하라.
# filter, lambda 함수 사용할 것
sample = [-3, -1.2, 0, 4, -0.5]

# 정답 41.
list(filter(lambda x: x>0, sample))

# 문제 42. 아래 세 개의 리스트를 원소 순서대로 묶어서 출력하라.
sample_1 = [3, 1, 2]
sample_2 = ['a', 'b', 'c']
sample_3 = {'a': 1, 'b': 2, 'c': 3}.values()

# 정답 42.
list(zip(sample_1, sample_2, sample_3))

# 문제 43. 아래 파일 목록에서 이미지 파일만 출력하라.
# 이미지 파일 확장자: '.png', '.jpg', '.gif'
# filter, lambda 함수 사용할 것

files = ['font', '1.png', '10.jpg', '11.gif', '2.jpg', '3.png', 'table.xslx', 'spec.docx']

# 정답 43.

# list(filter(lambda x: x.endswith(".png",".jpg,",".gif"), files))

print(list(filter(lambda x: x.endswith(".png"), files)))
print(list(filter(lambda x: x.endswith(".jpg"), files)))
print(list(filter(lambda x: x.endswith(".gif"), files)))

