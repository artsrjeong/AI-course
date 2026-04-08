import streamlit as st


st.set_page_config(page_title="Streamlit Cheat Sheet Style", layout="wide")

# Left sidebar
with st.sidebar:
    st.title("Streamlit")
    st.caption("API Cheat Sheet 스타일")
    st.markdown("---")
    st.subheader("Basics")
    st.markdown("- Text")
    st.markdown("- Data")
    st.markdown("- Charts")
    st.markdown("- Input Widgets")
    st.markdown("- Layouts")
    st.markdown("- Control Flow")
    st.markdown("---")
    st.subheader("About")
    st.write("Cheat-sheet 형식의 데모 화면입니다.")

# Right main area
st.title("Streamlit API Cheat Sheet")
st.caption("왼쪽은 내비게이션, 오른쪽은 3열 콘텐츠")
st.markdown("---")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.subheader("Text Elements")
    st.code('st.title("Title")', language="python")
    st.code('st.header("Header")', language="python")
    st.code('st.subheader("Subheader")', language="python")
    st.code('st.write("Hello Streamlit")', language="python")
    st.code('st.markdown("**Markdown**")', language="python")

    st.subheader("Input Widgets")
    st.code('st.button("Click")', language="python")
    st.code('st.checkbox("Check me")', language="python")
    st.code('st.radio("Choose", ["A", "B"])', language="python")

with col2:
    st.subheader("Data Elements")
    st.code("st.dataframe(df)", language="python")
    st.code("st.table(df)", language="python")
    st.code("st.metric('Sales', 1200, 5)", language="python")
    st.code("st.json({'name': 'Alice'})", language="python")

    st.subheader("Charts")
    st.code("st.line_chart(data)", language="python")
    st.code("st.bar_chart(data)", language="python")
    st.code("st.area_chart(data)", language="python")

with col3:
    st.subheader("Media & Layout")
    st.code('st.image("image.png")', language="python")
    st.code('st.audio("audio.mp3")', language="python")
    st.code('st.video("video.mp4")', language="python")
    st.code("st.columns(2)", language="python")
    st.code("st.tabs(['Tab1', 'Tab2'])", language="python")
    st.code('st.expander("Details")', language="python")

    st.subheader("Status")
    st.code('st.success("Done")', language="python")
    st.code('st.warning("Warning")', language="python")
    st.code('st.error("Error")', language="python")
