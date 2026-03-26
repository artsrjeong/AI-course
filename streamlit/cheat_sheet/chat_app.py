import streamlit as st
import numpy as np
# 입력 먼저 받기
prompt = st.chat_input("Say something")

# 입력값이 있으면 채팅 메시지로 출력
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
        st.line_chart(np.random.randn(30, 3))
else:
    with st.chat_message("user"):
        st.write("Hello 👋")
        st.line_chart(np.random.randn(30, 3))