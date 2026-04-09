import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Streamlit Cheat Sheet Style", layout="wide")

if "main_view" not in st.session_state:
    st.session_state.main_view = "cheat_sheet"


def _random_df() -> pd.DataFrame:
    rng = np.random.default_rng()
    return pd.DataFrame(
        {
            "항목": [f"항목 {i}" for i in range(1, 9)],
            "값_A": rng.integers(10, 100, size=8),
            "값_B": rng.normal(50, 15, size=8).round(2),
            "구분": rng.choice(["X", "Y", "Z"], size=8),
        }
    )


# Left sidebar
with st.sidebar:
    st.title("Streamlit")
    st.caption("Component 스타일")
    st.markdown("---")
    st.subheader("Basics")
    row_text_l, row_text_r = st.columns([1, 12], gap="small")
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Display Text", key="nav_text", use_container_width=True):
            st.session_state.main_view = "display_text"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Display Data", key="nav_data", use_container_width=True):
            st.session_state.main_view = "display_data"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Display Media", key="nav_media", use_container_width=True):
            st.session_state.main_view = "display_media"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Display Tab", key="nav_tab", use_container_width=True):
            st.session_state.main_view = "display_tab"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Display Control Flow", key="nav_control_flow", use_container_width=True):
            st.session_state.main_view = "control_flow"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Display Widget", key="nav_widget", use_container_width=True):
            st.session_state.main_view = "display_widget"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Chat App", key="nav_chat_app", use_container_width=True):
            st.session_state.main_view = "chat_app"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Mutate Data", key="nav_mutate_data", use_container_width=True):
            st.session_state.main_view = "mutate_data"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("Display Code", key="nav_display_code", use_container_width=True):
            st.session_state.main_view = "display_code"
    with row_text_l:
        st.markdown("-")
    with row_text_r:
        if st.button("PlaceHolder", key="nav_placeholder", use_container_width=True):
            st.session_state.main_view = "placeholder"


    st.markdown("---")
    st.subheader("About")
    st.write("Cheat-sheet 형식의 데모 화면입니다.")

# Right main area
st.title("Streamlit API Cheat Sheet")
st.caption("왼쪽은 내비게이션, 오른쪽은 Main Area")
st.markdown("---")

if st.session_state.main_view == "display_text":
    st.text('Fixed width text')
    st.markdown('_Markdown_') # see #*
    st.caption('Balloons. Hundreds of them...')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')
    st.write('Most objects') # df, err, func, keras!
    st.write(['st', 'is <', 3]) # see *
    st.title('My title')
    st.header('My header')
    st.subheader('My sub')
    st.code('for i in range(8): foo()')
if st.session_state.main_view == "text_df":
    st.subheader("Text — 임의 DataFrame")
    if "text_sample_df" not in st.session_state:
        st.session_state.text_sample_df = _random_df()
    st.dataframe(st.session_state.text_sample_df, use_container_width=True, hide_index=True)
else:
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
