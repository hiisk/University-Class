# 1번
i = 0
while True:
    i += 1
    if i > 5: break
    print('*' * i)

#2번
A = [70, 60, 55, 75, 95, 90, 80, 80, 85, 100]
total = 0
for score in A:
    total += score
average = total/10
print(average)

#3번
A = "881120-1068234"
b = ""
for i in A:
    if i == '-':
        break
    b += str(i)
print(b)