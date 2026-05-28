# 영어, 수학 순서로 튜플 데이터 저장
scores = [
    (80, 100),
    (100, 50),
    (70, 100),
    (80, 90)
]

# 영어 ↓, 영어가 같으면 수학 ↓
scores.sort(key=lambda x: (-x[0], -x[1]))

for s in scores:
    print(f"english={s[0]}, math={s[1]}")
