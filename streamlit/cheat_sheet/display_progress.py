import streamlit as st
import time

# Show a spinner during a process
with st.spinner(text='In progress'):
  time.sleep(3)
  st.success('Done')

# Show and update progress bar
bar = st.progress(50)
time.sleep(3)
bar.progress(100)

st.balloons()
st.snow()
st.toast('Mr Stay-Puft')
st.error('Error message')
st.warning('Warning message')
st.info('Info message')
st.success('Success message')

    