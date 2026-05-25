import random

testcase=int(input())
answer=0
A=[0]*(100001)

for i in range(0,10001):
    A[i]=random.randrange(1,101)

for t in range(1,testcase+1):
    start,end=map(int,input().split())

    for i in range(start,end+1):
        answer+=A[i]
    print(str(testcase)+" "+str(answer/2))

