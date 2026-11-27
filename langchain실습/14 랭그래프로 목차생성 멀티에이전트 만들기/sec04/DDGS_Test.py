from ddgs import DDGS


def main():
    query = input("검색어를 입력하세요 (기본값: Python programming language): ").strip()
    if not query:
        query = "Python programming language"

    print(f"DDGS 검색어: {query!r}")

    try:
        search_results = DDGS().text(query, max_results=5)
    except Exception as exc:
        print(f"DDGS 검색 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        raise

    if not search_results:
        print("검색 결과가 없습니다.")
        return

    for index, result in enumerate(search_results, start=1):
        print(f"\n[{index}] {result.get('title', '')}")
        print(f"URL: {result.get('href', '')}")
        print(result.get("body", ""))


if __name__ == "__main__":
    main()
