import streamlit as st
import pandas as pd
import numpy as np
n=15
# x는 1부터 n까지 증가
x = np.arange(1, n+1)
# y는 랜덤 숫자 (예: 0~1 사이)
y = np.random.rand(n)
# DataFrame 생성
df1 = pd.DataFrame({"x": x, "y": y})
df2 = pd.DataFrame({"x": x, "y": y})

# Add rows to a dataframe after
# showing it.
element = st.dataframe(df1)
element.add_rows(df2)

# Add rows to a chart after
# showing it.
element = st.line_chart(df1)
element.add_rows(df2)