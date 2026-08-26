import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 가상 데이터 생성 (다운로드 없이 즉시 실행)
# 1000개의 샘플, 10개의 특성(Feature), 이진 분류 문제
np.random.seed(42)
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)

# 2. [모델 1] 기본 모델 (Base Model) 생성 및 사전 학습
# 이미지 대신 1D 신경망으로 구조를 극단적으로 단순화합니다.
base_model = models.Sequential(
    [
        layers.Dense(32, activation="relu", input_shape=(10,)),
        layers.Dense(16, activation="relu", name="feature_extractor"),
    ],
    name="Base_Model",
)

base_model.compile(optimizer="adam", loss="mse")
print("=== [Step 1] Base Model 사전 학습 ===")
base_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

# 3. [개념 핵심] Base Model 가중치 동결 (Freezing)
base_model.trainable = False

print("\n--- 동결 확인 ---")
print(f"Base Model 학습 가능 파라미터 수: {base_model.count_params()}")
# trainable=False 이후 compile을 다시 하거나 상위 모델에 얹으면 가중치가 고정됩니다.


# 4. [모델 2] 전이 학습 모델 (Transfer Learning Model) 구축
# 동결된 Base Model 뒤에 새로운 Top Layer(분류기)를 붙입니다.
transfer_model = models.Sequential(
    [
        base_model,  # 사전 학습 및 동결된 모델
        layers.Dense(8, activation="relu"),  # 추가 레이어
        layers.Dense(1, activation="sigmoid"),  # 최종 출력 레이어
    ],
    name="Transfer_Model",
)

transfer_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# 5. 모델 요약으로 파라미터 변화 확인
print("\n=== [Step 2] 전이 학습 모델 구조 ===")
transfer_model.summary()

# 6. 전이 학습 진행
print("\n=== [Step 3] 동결된 상태로 추가 헤드만 학습 ===")
transfer_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)