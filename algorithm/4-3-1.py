N=int(input())
Result=0
A=list(map(int,input().split()))
A.sort()
for k in range(N):
    find=A[k]
    i=0
    j=N-1
    while i<j:
        if A[i]+A[j]==find:
            if i!=k and j!=k:
                Result+=1
                break
        elif A[i]+A[j]<find:
            i+=1
        else:
            j-=1
print(Result)