import streamlit as st
import numpy as np
import time

# --- 설정 및 환경 초기화 ---
st.set_page_config(page_title="RL Grid World Demo", layout="centered")
st.markdown('<h1 style="font-size: 1.8rem;">🤖 강화학습(Q-Learning) 미로 찾기 데모</h1>', unsafe_allow_html=True)

# 격자 크기 및 상태 설정
grid_size = 4
actions = ['Up', 'Down', 'Left', 'Right']
n_actions = len(actions)

# 상태 정의 (0: 빈칸, 1: 벽, 2: 함정, 3: 목적지)
grid = np.zeros((grid_size, grid_size))
grid[1, 1] = 1  # 벽
grid[2, 3] = 2  # 함정
grid[3, 3] = 3  # 목적지

# Q-Table 초기화 (상태 개수 x 행동 개수)
if 'q_table' not in st.session_state:
    st.session_state.q_table = np.zeros((grid_size * grid_size, n_actions))

def get_state_idx(r, c):
    return r * grid_size + c

# --- 학습 로직 (Q-Learning) ---
def train_agent(episodes=100, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    q_table = st.session_state.q_table
    
    for _ in range(episodes):
        r, c = 0, 0  # 시작점 (0,0)
        while grid[r, c] == 0:  # 목적지나 함정에 도달할 때까지
            state_idx = get_state_idx(r, c)
            
            # Epsilon-greedy 정책
            if np.random.uniform(0, 1) < epsilon:
                action_idx = np.random.randint(n_actions)
            else:
                action_idx = np.argmax(q_table[state_idx])
            
            # 행동 수행
            next_r, next_c = r, c
            if action_idx == 0 and r > 0: next_r -= 1    # Up
            elif action_idx == 1 and r < 3: next_r += 1  # Down
            elif action_idx == 2 and c > 0: next_c -= 1  # Left
            elif action_idx == 3 and c < 3: next_c += 1  # Right
            
            # 벽 충돌 처리
            if grid[next_r, next_c] == 1:
                next_r, next_c = r, c
            
            # 보상 설정
            if grid[next_r, next_c] == 3: reward = 10   # 목적지
            elif grid[next_r, next_c] == 2: reward = -10 # 함정
            else: reward = -1                            # 이동 페널티
            
            # Q-Value 업데이트 (Bellman Equation)
            next_state_idx = get_state_idx(next_r, next_c)
            best_next_q = np.max(q_table[next_state_idx])
            q_table[state_idx, action_idx] += learning_rate * (reward + discount_factor * best_next_q - q_table[state_idx, action_idx])
            
            r, c = next_r, next_c
            if grid[r, c] != 0: break # 종료 조건

# --- UI 레이아웃 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎮 환경 (Environment)")
    display_grid = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            if r == 0 and c == 0: row.append("🏠")
            elif grid[r, c] == 1: row.append("🧱")
            elif grid[r, c] == 2: row.append("💀")
            elif grid[r, c] == 3: row.append("🎁")
            else: row.append("⬜")
        display_grid.append(row)
    st.table(display_grid)

    if st.button("🚀 100회 학습하기"):
        with st.spinner('학습 중...'):
            train_agent(100)
        st.success("학습 완료!")

with col2:
    st.subheader("📊 Q-Table (지식)")
    st.dataframe(st.session_state.q_table, height=300)

# --- 결과 시각화 (Policy) ---
st.divider()
st.subheader("📍 학습된 최적 경로 (Policy)")
policy_grid = []
for r in range(grid_size):
    row = []
    for c in range(grid_size):
        if grid[r, c] == 1: row.append("🧱")
        elif grid[r, c] == 2: row.append("💀")
        elif grid[r, c] == 3: row.append("🎁")
        else:
            best_action = np.argmax(st.session_state.q_table[get_state_idx(r, c)])
            row.append(["↑", "↓", "←", "→"][best_action])
    policy_grid.append(row)

st.table(policy_grid)