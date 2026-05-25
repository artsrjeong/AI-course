# 원본 리스트 선언
A = [5, 3, 2, 4, 1]

# 방법 1: list.sort() - 리스트 자체를 정렬(in-place)
A.sort()    # sort() 오름차순 정렬
print("sort() 결과:", A)

# 원본 리스트 다시 초기화
A = [5, 3, 2, 4, 1]

# 방법 2: sorted() - 정렬된 복사본을 반환(원본 유지)
B = sorted(A)    # sorted() 오름차순 정렬
print("원본 리스트:", A)
print("sorted() 결과:", B)
