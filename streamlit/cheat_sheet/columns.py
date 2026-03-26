import streamlit as st
col1, col2 = st.columns(2)
col1.write('Column 1')
col2.write('Column 2')

# Three columns with different widths
col1, col2, col3 = st.columns([3,1,1])
# col1 is wider
              
# Using 'with' notation:
with col1:
    st.write('This is column 1')
with col2:
    st.write('This is column 2')
with col3:
    st.write('This is column 3')
              