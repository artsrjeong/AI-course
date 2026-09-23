from ddgs import DDGS
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from datetime import datetime

import json
import os
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로

@tool
def web_search(query: str):
    """
    주어진 query에 대해 웹검색을 하고, 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        dict: 검색 결과
    """
    search_results = DDGS().text(query, max_results=5)
    results = [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "content": result.get("body", ""),
            "raw_content": None,
        }
        for result in search_results
    ]   #②

    for result in results:
        if result["raw_content"] is None:
            try:
                result["raw_content"] = load_web_page(result["url"])
            except Exception as e:
                print(f"Error loading page: {result['url']}")
                print(e)
                result["raw_content"] = result["content"]

    data_path = os.path.join(current_path, "data")
    os.makedirs(data_path, exist_ok=True)
    resources_json_path = os.path.join(
        data_path,
        f"resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json",
    )
    with open(resources_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
   
    return results, resources_json_path  #③ (3) 검색 결과와 JSON 파일 경로 반환



def load_web_page(url: str):
    loader = WebBaseLoader(url, verify_ssl=False)

    content = loader.load()
    raw_content = content[0].page_content.strip()   #①

    # PDF 바이너리가 HTML 본문처럼 읽히는 경우 검색 요약을 사용합니다.
    if raw_content.startswith("%PDF") or "stream" in raw_content[:1000]:
        raise ValueError("PDF content cannot be loaded with WebBaseLoader")

    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n')
        raw_content = raw_content.replace('\t\t\t', '\t\t')
        
    return raw_content

if __name__ == "__main__":
    results, resources_json_path = web_search.invoke("2025년 한국 경제 전망")
    print(results)

    # result = load_web_page("https://eiec.kdi.re.kr/publish/columnView.do?cidx=15029&ccode=&pp=20&pg=&sel_year=2025&sel_month=01")
    # print(result)

