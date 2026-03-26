import streamlit as st
import seaborn as sns
data = sns.load_dataset("iris")

st.button('Hit me')
st.data_editor(data)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')
# DataFrame을 CSV로 변환
csv = data.to_csv(index=False).encode('utf-8')
# 다운로드 버튼
st.download_button(
    label="download data",
    data=csv,
    file_name="data.csv",
    mime="text/csv"
)
#st.download_button('On the dl', data)
st.camera_input("一二三,茄子!")
st.color_picker('Pick a color')
    