import streamlit as st
import pandas as pd
import numpy as np

# Replace any single element.
df=pd.DataFrame(np.random.randn(30,3),columns=['A','B','C'])
element = st.empty()
element.line_chart(df)
element.text_input("Enter your name", key="input_1")  # Replaces previous.

# Insert out of order.
elements = st.container()
elements.line_chart(df)
st.write("Hello")
elements.text_input("Enter your name", key="input_2")  # Appears above "Hello".

st.help(df)
# 아래 코드들은 사용법을 보여주기 위한 예시(Cheat Sheet)이므로 주석 처리합니다.
st.get_option('theme.base')  # 예: 'theme.base' 등의 문자열 키가 들어갑니다.
#st.set_option('theme.base', 'light') 
st.set_page_config(layout='wide')  # (주의) set_page_config는 반드시 스크립트 최상단에 위치해야 합니다.
    