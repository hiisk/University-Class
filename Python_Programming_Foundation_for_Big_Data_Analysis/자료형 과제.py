# 1번
pin = "881120-1068234"
yyyymmdd = pin[:6]
num = pin[7:]
print(yyyymmdd)
print(num)

# 2번
pin = "881120-1068234"
print(pin[7])

# 3번
a = 3
b = 4

if a < b:
    print("a is less than b")
if a > b:
    print("a is greater than b")
if a == b:
    print("a is equal to b")

# 4번

score = 85
if 90 <= score:
    print("당신의 점수는 {}점이므로, 성적은 A입니다".format(score))
elif 80 <= score <= 89:
    print("당신의 점수는 {}점이므로, 성적은 B입니다".format(score))
elif 70 <= score <= 79:
    print("당신의 점수는 {}점이므로, 성적은 C입니다".format(score))
elif 60 <= score <= 69:
    print("당신의 점수는 {}점이므로, 성적은 D입니다".format(score))
else:
    print("당신의 점수는 {}점이므로, 성적은 F입니다".format(score))