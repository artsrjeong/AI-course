import time

answer = 1
start = time.time()

for i in range(1, 100001):
    #answer *= i
    answer = (answer * i) % 1000000007

result = answer % 1000000007
end = time.time()
print("결과:", result)
print("수행 시간: {:.6f}초".format(end - start))
