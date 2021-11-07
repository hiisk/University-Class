#1번
def fib(n):
    if n==0:
        return 0
    if n==1:
        return 1
    return fib(n-1)+fib(n-2)

n = int(input("정수를 입력하세요: "))

for i in range(n):
    print (fib(i))

#2번
def list_size_check(a,b):
    if a==b:
        print(True)
        return True
    if a!=b:
        print(False)
        return False

list1 = [1, 2, 3, 4]
list2 = [1, 2, 3, 4, 5]
list_size_check(list1, list2)

#3번
result = [x * y for x in range(2, 10) for y in range(1, 10)]
print(result)