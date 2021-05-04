import time
global a
a = 1
def ddd():
    global a
    a=a+1
    print(a)
    a=4
ddd()

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))