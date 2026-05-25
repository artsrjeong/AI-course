# AI Coding Instructions: Add News Aggregator MCP Tool

## 🎯 Goal
`mcp/main.py` 파일 내에 `playwright`를 활용하여 최근 1주일 동안 발생한 주요 뉴스를 크롤링하고 요약해 주는 새로운 `mcp.tool`을 추가합니다.

## 🛠️ Tech Stack & Requirements
- **Framework:** FastMCP (Python)
- **Scraping Library:** `playwright` (Async API 권장)
- **Target File:** `mcp/main.py`

## 📋 Detailed Instructions for AI

### 1. 의존성 확인 및 모듈 임포트
- 스크립트 상단에 `playwright.async_api` 및 날짜 계산을 위한 `datetime`, `timedelta`가 임포트되어 있는지 확인하고 없으면 추가하세요.
- 필요시 비동기 처리를 위한 `asyncio` 혹은 BeautifulSoup(`bs4`)이 조합될 수 있습니다.

### 2. `mcp.tool` 등록 가이드라인
- **데코레이터:** `@mcp.tool()`을 사용하여 함수를 등록하세요.
- **함수명:** `get_weekly_top_news`로 지정하세요.
- **Docstring:** LLM이 도구의 목적을 완벽히 이해할 수 있도록 상세한 한글 Docstring을 작성하세요.
- **인자(Arguments):** 
  - `keyword: str = None` (선택 사항: 특정 키워드 뉴스 검색용)

### 3. Playwright 크롤링 로직 디자인
- **대상 사이트:** 네이버 뉴스(종합/랭킹) 혹은 구글 뉴스 RSS/검색 등 데이터 수집이 용이하고 신뢰할 수 있는 뉴스 플랫폼을 타겟팅하세요.
- **날짜 필터링:** 현재 날짜(`datetime.now()`) 기준으로 **최근 1주일(7일 전 ~ 현재)** 사이에 발행된 뉴스만 필터링 또는 수집되도록 검색 쿼리 매개변수를 활용하거나 날짜 요소를 파싱하세요.
- **헤드리스 모드:** Playwright 실행 시 `headless=True`로 설정하여 백그라운드에서 동작하게 하세요.

### 4. 출력 포맷 (Output Schema)
- 수집된 뉴스는 구조화된 마크다운 문자열로 반환되어야 합니다.
- **반환 포맷 예시:**
  ```markdown
  ### 📰 최근 1주일 주요 뉴스 브리핑 (기간: YYYY-MM-DD ~ YYYY-MM-DD)

  1. **[정치/경제/IT 등 카테고리] 뉴스 제목**
     - **링크:** [기사 보기](URL)
     - **요약:** 크롤링한 헤드라인 및 본문 핵심 내용 요약 (1~2문장)
  ```

### 5. 에러 핸들링 및 안정성
- 웹 크롤링 시 발생할 수 있는 타임아웃, 네트워크 에러, DOM 구조 변경에 대비해 `try-except` 문을 구성하세요.
- 예외 발생 시 크래시를 내지 않고, LLM이 식별할 수 있는 명확한 에러 메시지를 마크다운 형태로 반환하세요.

## 💻 구현 코드 예시 참고 (Reference Python Layout)
AI는 기존 `mcp/main.py` 파일의 구조를 깨뜨리지 않는 선에서 아래와 같은 구조의 코드를 삽입해야 합니다.

```python
from playwright.async_api import async_playwright
from datetime import datetime, timedelta
# 기존 임포트 유지...

@mcp.tool()
async def get_weekly_top_news(keyword: str = None) -> str:
    """
    최근 1주일 동안 발생한 주요 뉴스를 Playwright를 통해 수집하고 요약하여 반환합니다.
    keyword 인자가 주어지면 해당 키워드와 관련된 뉴스를 검색합니다.
    """
    # 1. 날짜 범위 계산 (최근 7일)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # 2. Playwright 크롤링 로직 수행
    # (여기에 비동기 브라우저 실행 및 셀렉터 추출 로직 구현)
    
    # 3. 마크다운 포맷팅 후 return
    return "결과 마크다운 문자열"
```

## 🚨 확인 및 마무리 작업
- 코드 추가 후, 사용자가 개발 환경에서 `playwright install`을 실행해야 할 수도 있음을 주석이나 안내 메시지로 남겨주세요.
