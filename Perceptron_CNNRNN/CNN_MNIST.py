import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def main():
    # 1. 데이터 준비
    num_classes = 10
    input_shape = (28, 28, 1)

    # MNIST 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 처음 10개의 이미지를 화면에 출력하여 확인
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 이미지 픽셀 값을 0 ~ 1 사이로 정규화
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # 차원 추가 (샘플 수, 28, 28) -> (샘플 수, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # 레이블을 원-핫 인코딩(One-hot encoding)으로 변환
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # 2. CNN 모델 구성
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    # 3. 모델 학습 설정 및 학습 진행
    batch_size = 128
    epochs = 5

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print("\n모델 학습을 시작합니다...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # 4. 모델 평가
    score = model.evaluate(x_test, y_test, verbose=0)
    print("\n--- 학습 결과 ---")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

if __name__ == "__main__":
    main()
