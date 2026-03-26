import streamlit as st

# Stop execution immediately:
#st.stop()
# Rerun script immediately:
#st.rerun()


# Group multiple widgets:
with st.form(key='my_form'):
    username = st.text_input('Username')
    password = st.text_input('Password')
    st.form_submit_button('Login')