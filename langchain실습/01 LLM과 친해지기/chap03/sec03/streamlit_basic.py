import streamlit as st
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# (0) 환경변수 또는 사이드바에서 GEMINI_API_KEY 불러오기
with st.sidebar:
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    # gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
    "[Get a Gemini API key](https://aistudio.google.com/app/apikey)"

st.title("💬 Chatbot")

# (1) st.session_state에 "messages"가 없으면 초기값 설정
# 주의: google-genai SDK의 대화 히스토리 형식은 "user"와 "model"을 사용합니다.
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "model", "content": "How can I help you?"}]

# (2) 대화 기록을 화면에 출력
for msg in st.session_state["messages"]:
    # UI 표시용 역할 변환 ("model" -> "assistant")
    display_role = "assistant" if msg["role"] == "model" else msg["role"]
    st.chat_message(display_role).write(msg["content"])

# (3) 사용자 입력을 받아 대화 기록에 추가하고 AI 응답 생성
if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("Please add your Gemini API key to continue.")
        st.stop()

    # Gemini 클라이언트 생성
    client = genai.Client(api_key=gemini_api_key)

    # 사용자 메시지 추가 및 화면 출력
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Gemini 대화 세션 생성 및 이전 메시지 복원
    # history 형식: [{'role': 'user', 'parts': [{'text': '...'}]}, ...]
    formatted_history = [
        {"role": msg["role"], "parts": [{"text": msg["content"]}]}
        for msg in st.session_state["messages"][:-1]
    ]

    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        history=formatted_history
    )

    # 응답 생성
    response = chat.send_message(prompt)
    msg = response.text

    # 응답 저장 및 화면 출력
    st.session_state["messages"].append({"role": "model", "content": msg})
    st.chat_message("assistant").write(msg)