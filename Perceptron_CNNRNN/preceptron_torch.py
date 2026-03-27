import torch
import torch.nn as nn
import torch.optim as optim

# 1. 훈련 데이터 준비 (AND 게이트)
# 입력 데이터: 2개의 특성 (input: 2)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
# 정답 데이터 (타겟): 1개의 출력 (output: 1)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# 2. 모델 구성
# 단일 퍼셉트론 구현: 입력 2개, 출력 1개, 활성화 함수로 시그모이드(sigmoid) 사용
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        # 입력 차원 2, 출력 차원 1인 선형 변환
        self.linear = nn.Linear(2, 1)
        # 시그모이드 활성화 함수
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = Perceptron()

# 3. 모델 손실 함수(Loss) 및 최적화(Optimizer) 설정
# 이진 분류 문제이므로 BCELoss (Binary Cross Entropy Loss) 사용
criterion = nn.BCELoss()
# 최적화는 확률적 경사 하강법(SGD) 사용
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 모델 학습
print("모델 학습을 시작합니다. (1000 에포크)")
epochs = 1000
for epoch in range(epochs):
    # Forward pass: 입력 X를 모델에 통과시켜 예측값 계산
    outputs = model(X)
    
    # Loss 계산
    loss = criterion(outputs, y)
    
    # Backward pass 및 가중치 업데이트
    optimizer.zero_grad() # 이전 기울기 초기화
    loss.backward()       # 역전파로 기울기 계산
    optimizer.step()      # 계산된 기울기로 가중치 업데이트

print("학습이 완료되었습니다.\n")

# 5. 모델 평가 및 예측 결과 출력
print("--- AND 게이트 예측 결과 ---")
with torch.no_grad(): # 평가 혹은 예측 시에는 기울기 계산(역전파)을 비활성화
    predictions = model(X)
    
    for i in range(len(X)):
        # 예측값이 0.5 이상이면 1, 아니면 0으로 분류
        pred_prob = predictions[i].item()
        predicted_class = 1 if pred_prob >= 0.5 else 0
        input_data = [int(v.item()) for v in X[i]]
        print(f"입력: {input_data} => 예측 확률: {pred_prob:.4f}, 예측 클래스: {predicted_class}")
