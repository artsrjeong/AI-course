import numpy as np

# 활성화 함수 (시그모이드 함수)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분 (역전파시 기울기 계산에 사용)
def sigmoid_derivative(x):
    return x * (1 - x)

# 1. 데이터 준비 (XOR 문제)
# 입력 데이터: [x1, x2]
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# 정답(Target) 데이터 (XOR 결과)
y = np.array([[0],
              [1],
              [1],
              [0]])

# 실험의 재현성을 위해 시드 고정
np.random.seed(42)

# 2. 가중치(Weights)와 편향(Biases) 초기화
# 은닉층 노드는 2개이므로 (입력층 노드 2 -> 은닉층 노드 2) 개수만큼 가중치와 편향 생성
# (이 두 개의 은닉 노드가 각각 NAND와 OR의 역할을 하도록 학습될 것입니다)
W1 = np.random.uniform(-1, 1, size=(2, 2))
b1 = np.random.uniform(-1, 1, size=(1, 2))

# 출력층 노드는 1개이므로 (은닉층 노드 2 -> 출력층 노드 1) 개수만큼 가중치와 편향 생성
# (은닉층의 두 출력을 취합받아 AND 연산 형태로 최종 결과를 만듭니다)
W2 = np.random.uniform(-1, 1, size=(2, 1))
b2 = np.random.uniform(-1, 1, size=(1, 1))

# 하이퍼파라미터 설정
epochs = 20000        # 학습(반복) 횟수
learning_rate = 0.5   # 학습률

print("=========================")
print("  XOR 신경망 학습 시작   ")
print("=========================")

# 3. 모델 학습 (순전파 및 역전파)
for epoch in range(epochs + 1):
    
    # [순전파 (Forward Propagation)]
    # 은닉층 (Hidden Layer): 입력값에 가중치를 곱하고 편향을 더한 후 활성화 함수 통과
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # 출력층 (Output Layer): 은닉층 결과를 입력으로 받아 동일하게 처리
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)
    
    # [오차 계산]
    error = y - predicted_output
    
    # [역전파 (Backpropagation)]
    # 출력층에서의 기울기 계산
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # 은닉층으로 오차 역전파 및 은닉층 기울기 계산
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # [가중치 및 편향 업데이트]
    W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # 진행 상황 출력 (2000 에포크마다)
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch: {epoch:>6} | 손실(Loss): {loss:.5f}")


print("\n=========================")
print("       학습 완료!        ")
print("=========================\n")

# 4. 결과 확인
print("1. 입력 X:")
print(X)

print("\n2. 은닉층의 결과값 (NAND, OR 게이트 역할 지표):")
# 보통 하나는 NAND, 하나는 OR 형태와 근사한 출력을 뱉도록 최적화됩니다.
print(np.round(hidden_layer_output, 3))

print("\n3. 최종 출력 결과값 (AND 게이트 결합 효과 / 결과 확인):")
print(predicted_output)

print("\n4. 0과 1로 반올림한 최종 예측값:")
print(np.round(predicted_output))
