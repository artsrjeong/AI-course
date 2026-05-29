import sqlite3
import seaborn as sns


def main():
    print("1. Seaborn에서 'iris' 데이터셋 로드 중...")
    # 파이썬에서 제공되는 기본 데이터셋(DataFrame) 로드
    df = sns.load_dataset("iris")

    # 데이터셋 구조 확인을 위한 출력
    print("\n[데이터셋 상위 5개 행 미리보기]")
    print(df.head())

    # 2. SQLite DB 연결 (파일이 없으면 자동 생성됨)
    db_path = "iris_data.db"
    conn = sqlite3.connect(db_path)

    print(f"\n2. '{db_path}' 데이터베이스에 연결 성공.")

    # 3. DataFrame을 SQL 테이블로 저장
    # name: 테이블명, if_exists: 기존 테이블이 있으면 덮어쓰기(replace)
    table_name = "iris_table"
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    # 연결 종료
    conn.close()
    print(f"3. '{table_name}' 테이블에 데이터 저장 완료 및 DB 연결 종료!")


if __name__ == "__main__":
    main()

