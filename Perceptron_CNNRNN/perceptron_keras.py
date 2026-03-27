import numpy as np
import tensorflow as tf
from tensorflow import keras

# 1. 훈련 데이터 준비 (AND 게이트)
# 입력 데이터: 2개의 특성 (input: 2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
# 정답 데이터 (타겟): 1개의 출력 (output: 1)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# 2. 모델 구성
# 단일 퍼셉트론 구현: 입력 2개, 출력 1개, 활성화 함수로 시그모이드(sigmoid) 사용
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(2,), activation='sigmoid')
])

# 3. 모델 컴파일
# 이진 분류 문제이므로 loss로 binary_crossentropy 사용, 최적화는 확률적 경사 하강법(SGD) 사용
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. 모델 학습
print("모델 학습을 시작합니다. (1000 에포크)")
model.fit(X, y, epochs=1000, verbose=0) # 학습 과정 출력을 생략(verbose=0)
print("학습이 완료되었습니다.\n")

# 5. 모델 평가 및 예측 결과 출력
print("--- AND 게이트 예측 결과 ---")
predictions = model.predict(X, verbose=0)

for i in range(len(X)):
    # 예측값이 0.5 이상이면 1, 아니면 0으로 분류
    predicted_class = 1 if predictions[i][0] >= 0.5 else 0
    print(f"입력: {X[i].astype(int)} => 예측 확률: {predictions[i][0]:.4f}, 예측 클래스: {predicted_class}")
