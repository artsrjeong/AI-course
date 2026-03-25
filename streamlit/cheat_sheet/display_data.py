import streamlit as st
import pandas as pd
import seaborn as sns
data = sns.load_dataset("iris")
print(data.head())
my_dataframe = pd.DataFrame({'col1': [1,2,3]})
st.dataframe(my_dataframe)
st.table(data.iloc[0:10])
st.json({'foo':'bar','fu':'ba'})
st.metric(label="Temp", value="273 K", delta="1.2 K")
    