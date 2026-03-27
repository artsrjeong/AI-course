import random
import math

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        """
        퍼셉트론 초기화
        :param input_size: 입력 데이터의 크기 (여기서는 2)
        :param learning_rate: 학습률
        """
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """시그모이드 활성화 함수"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1

    def sigmoid_derivative(self, x):
        """시그모이드 함수의 미분값 (경사하강법 계산용)"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def predict(self, inputs):
        """출력 예측 (순전파)"""
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sigmoid(weighted_sum)

    def train(self, training_data, labels, epochs=10000):
        """
        경사하강법을 이용한 학습
        :param training_data: 입력 데이터 리스트
        :param labels: 정답 레이블 리스트
        :param epochs: 학습 횟수
        """
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                # 1. 순전파
                weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
                prediction = self.sigmoid(weighted_sum)
                
                # 2. 오차 계산 (MSE 기준: (prediction - label)^2 / 2)
                error = prediction - label
                total_error += error**2

                # 3. 경사 계산 및 가중치 업데이트 (경사하강법)
                # d(Error)/d(Weight) = (prediction - label) * sigmoid'(weighted_sum) * input
                # d(Error)/d(Bias) = (prediction - label) * sigmoid'(weighted_sum)
                delta = error * self.sigmoid_derivative(weighted_sum)
                
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * delta * inputs[i]
                self.bias -= self.learning_rate * delta

            # 1000회마다 오차 출력
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {total_error / len(training_data):.4f}")
                print(self.weights, self.bias)

if __name__ == "__main__":
    # AND 게이트 데이터셋 (2개 입력, 1개 출력)
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y = [0, 0, 0, 1]

    # 퍼셉트론 생성 및 학습
    print("--- 퍼셉트론 학습 시작 (AND Gate) ---")
    p = Perceptron(input_size=2, learning_rate=0.5)
    p.train(X, y, epochs=5000)

    print("\n--- 학습 결과 테스트 ---")
    for inputs in X:
        result = p.predict(inputs)
        print(f"입력: {inputs}, 예측값: {result:.4f} -> 결과: {1 if result >= 0.5 else 0}")
