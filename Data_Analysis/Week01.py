
# 문제 1. Hello, World! 출력하세요.

# 정답 1.
print('Hello, World!')

# 문제 2. samsung 이라는 변수로 50000원을 설정하시오. samsung 주식 10주를 보유하고 있을 때, 총 평가금액을 출력하세요.

# 정답 2.
samsung = 50000
print(samsung * 10)

# 문제 3. 아래 변수에 바인딩된 값의 타입을 판별해보세요.
a = "132"

# 정답 3.
type(a)

# 문제 4. 문자열 "720"을 정수형으로 변환한 변수와 타입을 출력하세요.
num_str = "720"

# 정답 4.
num_str = int(num_str)
print(num_str, type(num_str))

# 문제 5. 실수형 3.141592 를 문자열로 변환한 변수와 타입을 출력하세요.
pi = 3.141592

# 정답 5.
pi = str(pi)
print(pi, type(pi))

# 문제 6. 문자열 "3.141592" 를 실수형으로 변환한 변수와 타입을 출력하세요.
pi = "3.141592"

# 정답 6.
pi = float(pi)
print(pi, type(pi))

# 문제 7. 에어컨이 월 48,584원에 무이자 36개월의 조건으로 홈쇼핑에서 판매 중입니다. 총 구입 금액을 계산한 후 이를 출력하세요.

# 정답 7.
price_month = 48584
total_price = price_month * 36
print(total_price)

# 문제 8. 두 정수 A와 B를 입력받은 다음, A+B 결과값을 출력하는 프로그램을 작성하시오.
# 입력 시, A에 30, B에 10 입력하세요.
# 문제 링크: https://www.acmicpc.net/problem/1000

# 정답 8.
values = input()
values = values.split()
a = int(values[0])
b = int(values[1])
# 코드 작성: A+B 결과값 출력하는 코드
print(a+b)

# 문제 9. 두 정수 A와 B를 입력받은 다음, A/B 결과값을 출력하는 프로그램을 작성하시오.
# 입력 시, A에 30, B에 10 입력하세요.
# 문제 링크: https://www.acmicpc.net/problem/1008

# 정답 9.
values = input()
values = values.split()
a = int(values[0])
b = int(values[1])
# 코드 작성: A/B 결과값 출력하는 코드
print(a / b)

# 문제 10. 학번과 이름이 다음과 같을 때 이름만 출력하기
license_plate = "20187100 강현석"  # 수강생 분의 학번, 이름 반영 바랍니다.

# 정답 10.
name = license_plate[-3:]
print(name)

# 문제 11. 아래의 전화번호에서 하이픈('-')을 제거하고 출력하기
phone_number = "010-1111-2222"

# 정답 11.
phone_number_1 = phone_number.replace('-','')
print(phone_number_1)

# 문제 12. 아래 문자열에서 소문자 'a'를 대문자 'A'로 변경하기
string = "abcdfe2a354a32"

# 정답 12.
string = string.replace('a','A')
print(string)

# 문제 13. 문자열을 소문자 btc_krw로 변경하기
ticker = "BTC_KRW"

# 정답 13.
ticker = ticker.lower()
print(ticker)

# 문제 14. 파일 이름이 문자열로 저장되어 있을 때, startswith 함수로
# 파일 이름이 '2020'로 시작하는지 확인하기
file_name = "2020_보고서.xlsx"

#정답 14.
file_name.startswith('2020')

# 문제 15. 아래 문자열을 endswith 함수로 파일 이름이 'xlsx'로 끝나는지 확인하기
file_name = "보고서.xlsx"

# 정답 15.
file_name.endswith('xlsx')

# 문제 16. 다음의 문자열을 btc, krw로 나누기
ticker = "btc_krw"

# 정답 16.
ticker.split('_')

# 문제 17. lang1과 lang2 리스트의 원소를 모두 갖는 langs 리스트 만들기
lang1 = ["C", "C++", "JAVA"]
lang2 = ["Python", "Go", "C#"]

# 정답 17.
langs = lang1 + lang2
print(langs)

# 문제 18. 다음 리스트에 저장된 데이터의 개수를 화면에 출력하기
cook = ["피자", "김밥", "만두", "양념치킨", "족발", "피자", "김치만두", "쫄면"]

# 정답 18.
len(cook)

# 문제 19. 슬라이싱 사용하여 홀수만 출력하기
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 정답 19.
print(nums[::2])

# 문제 20. 리스트에 아래의 데이터가 바인딩 되어 있다.
interest = ['삼성전자', 'LG전자', 'NAVER', 'SK하이닉스','미래에셋대우']
# 아래와 같이 화면에 출력하라.
# 출력 예시: 삼성전자/LG전자/NAVER/SK하이닉스/미래에셋대우

# 정답 20.
print('/'.join(interest))

# 문제 21. 다음 딕셔너리에서 메로나의 가격을 1300으로 수정하라.
ice = {
    '메로나': 1000,
    '폴로포': 1200,
    '빵빠레': 1800,
    '죠스바': 1200,
    '월드콘': 1500
}

# 정답 21.
ice['메로나'] = 1300
print(ice)

# 문제 22. 다음 딕셔너리에서 메로나를 삭제하라.
ice = {
    '메로나': 1000,
    '폴로포': 1200,
    '빵빠레': 1800,
    '죠스바': 1200,
    '월드콘': 1500
}

# 정답 22.
del ice['메로나']
print(ice)

# 문제 23. 다음 딕셔너리에서 메로나의 가격을 화면에 출력하라.
inventory = {
    "메로나": [300, 20],
    "비비빅": [400, 3],
    "죠스바": [250, 100]
}

# 정답 23.
print(inventory["메로나"][0])

"""연습문제"""

# 문제 24. 두 정수 A와 B를 입력받은 다음, A+B, A-B, A*B, A/B(몫), A%B(나머지) 정수형 결과값을 출력하는 프로그램을 작성하시오.
# 입력 시, A에 32, B에 3 입력하세요.
# 문제 링크: https://www.acmicpc.net/problem/10869

# 정답 24.
values = input()
values = values.split()
a = int(values[0])
b = int(values[1])
# 코드 작성: A+B 결과값 출력
# 코드 작성: A-B 결과값 출력
# 코드 작성: A*B 결과값 출력
# 코드 작성: A/B 결과값 출력
# 코드 작성: A%B 결과값 출력
print(a+b ,a-b ,a*b ,int(a/b) ,a%b)

# 문제 25. 최초로 이익이 발생하는 판매량(> 손익분기점)을 출력한다. 이익이 발생하지 않으면 -1을 출력한다.
# A는 고정비용, B는 가변 비용, C는 제품 가격이다.
# 입력 시 1000 70 170 입력하세요.
# 문제 링크: https://www.acmicpc.net/problem/1712
# 손익분기점 계산식: a + b * x = c * x

# 정답 25.
values = input()
values = values.split()
a = int(values[0])
b = int(values[1])
c = int(values[2])
# A는 고정비용, B는 가변 비용, C는 제품 가격이다.
# 손익분기점 계산식: a + b * x = c * x

# 코드 작성: 이익이 발생하지 않는 경우 -1 출력
# 코드 작성: 최초로 이익이 발생하는 정수형 판매량(> 손익분기점) 출력
if b>=c:
    print(-1)
else:
    print(int(a/(c-b)+1))

# 문제 26. 슬라이싱 사용하여 리스트의 숫자를 역 방향으로 출력하기
nums = [1, 2, 3, 4, 5]

# 정답 26.
nums_reverse = nums[-1::-1]
print(nums_reverse)

# 문제 27. 회사 이름이 슬래시 ('/')로 구분되어 하나의 문자열로 저장되어 있다.
string = "삼성전자/LG전자/NAVER"
# 이를 3개의 회사 이름으로 분리시키기

# 정답 27.
string.split('/')

# 문제 28. 다음 딕셔너리에서 key 값으로만 구성된 리스트를 생성하라.
icecream = {"탱크보이": 1200, "폴라포": 1200, "빵빠레": 1800, "월드콘": 1500, "메로나": 1000}

# 정답 28.
ice = list(icecream.keys())
print(ice)

# 문제 29. 다음 딕셔너리에서 values 값으로만 구성된 리스트를 생성하라.
icecream = {"탱크보이": 1200, "폴라포": 1200, "빵빠레": 1800, "월드콘": 1500, "메로나": 1000}

# 정답 29.
ice = list(icecream.values())
print(ice)

