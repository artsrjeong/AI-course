import sqlite3
import pandas as pd
import streamlit as st

# --- 웹 페이지 설정 ---
st.set_page_config(page_title="Iris DB 대시보드", layout="wide")

st.title("📊 SQLite 데이터베이스 연동 대시보드")
st.markdown("Seaborn의 Iris 데이터셋을 SQLite DB에서 읽어와 표시하는 화면입니다.")

# --- 데이터베이스 읽기 함수 ---


@st.cache_data  # 데이터를 캐싱하여 매번 DB에 접근하지 않도록 성능 최적화
def load_data_from_db():
    db_path = "../iris_data.db"
    table_name = "iris_table"

    # DB 연결 및 쿼리 실행
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()

    return df


# 데이터 로드
try:
    df = load_data_from_db()

    # --- 사이드바: 필터링 기능 ---
    st.sidebar.header("Filter Options")
    # 품종(species)별로 필터링할 수 있는 멀티 셀렉트 박스
    species_list = df["species"].unique().tolist()
    selected_species = st.sidebar.multiselect(
        "품종 선택", species_list, default=species_list
    )

    # 필터링 적용
    filtered_df = df[df["species"].isin(selected_species)]

    # --- 메인 화면 레이아웃 분할 ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 원본 데이터 (DataFrame)")
        # Streamlit의 데이터프레임 컴포넌트로 데이터 출력
        st.dataframe(filtered_df, use_container_width=True)

    with col2:
        st.subheader("📈 간단한 데이터 시각화")
        # 품종별 평균 수치 요약 테이블
        summary_df = (
            filtered_df.groupby("species")
            .mean(numeric_only=True)
            .reset_index()
        )
        st.write("품종별 평균 수치:")
        st.dataframe(summary_df, use_container_width=True)

        # 품종별 꽃받침 길이(sepal_length) 바 차트
        st.bar_chart(
            data=filtered_df,
            x="species",
            y="sepal_length",
            use_container_width=True,
        )

except Exception as e:
    st.error(
        f"데이터를 불러오는 중 오류가 발생했습니다. 먼저 'save_to_db.py'를 실행해 주세요. (오류 내용: {e})"
    )
