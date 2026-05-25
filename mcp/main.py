from mcp.server.fastmcp import FastMCP
import asyncio
from playwright.async_api import async_playwright
from datetime import datetime, timedelta

# Create the FastMCP instance with stdio transport
mcp = FastMCP()

# Define the tool using the @mcp.tool() decorator
@mcp.tool()
def get_weather(city: str) -> str:
   """
   Returns weather of the city

   :param city: The city to get the weather for
   """
   return f"{city} weather is good"

@mcp.tool()
async def get_du_notices():
    """동서울대학교 일반공지 게시판에서 최신 공지사항 목록을 가져옵니다."""
    url = "https://www.du.ac.kr/KR/cms/board/board.do?mCode=MN0033"
    
    async with async_playwright() as p:
        # 브라우저 실행 (headless=True 가 기본)
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle")
            
            # 공지사항 리스트 행(tr) 추출
            # 동서울대 사이트 구조에 맞춘 selector (일반적으로 tbody 내의 tr들)
            rows = await page.query_selector_all("table.board-table tbody tr")
            
            notices = []
            for row in rows:
                # 제목과 번호 추출
                title_el = await row.query_selector("td.subject a")
                date_el = await row.query_selector("td.date")
                
                if title_el:
                    title = (await title_el.inner_text()).strip()
                    date = (await date_el.inner_text()).strip() if date_el else "날짜 없음"
                    link = await title_el.get_attribute("href")
                    
                    notices.append(f"📌 {title} ({date})")
            
            if not notices:
                return "공지사항을 찾을 수 없습니다. 사이트 구조가 변경되었을 수 있습니다."
            
            return "\n".join(notices[:10])  # 최신 10개만 반환
            
        except Exception as e:
            return f"에러 발생: {str(e)}"
        finally:
            await browser.close()

@mcp.tool()
async def get_naver_sports_news():
    """네이버에서 최신 스포츠 뉴스 소식을 가져옵니다."""
    url = "https://sports.naver.com/"
    
    async with async_playwright() as p:
        # headless=True(기본값)로 실행
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle")
            
            # 네이버 스포츠 메인에서 기사 제목(클래스에 title이 포함된 요소들) 추출
            title_elements = await page.query_selector_all(".title")
            
            news_list = []
            for el in title_elements:
                title = (await el.inner_text()).strip()
                if title and title not in news_list:
                    news_list.append(title)
            
            # 제목을 찾지 못했을 경우 백업 플랜: 텍스트 길이가 15자 이상인 a 태그 추출
            if not news_list:
                links = await page.query_selector_all("a")
                for link in links:
                    text = (await link.inner_text()).strip()
                    if text and len(text) > 15 and text not in news_list:
                        news_list.append(text)
                        
            if not news_list:
                return "스포츠 뉴스를 찾을 수 없습니다. 사이트 구조가 변경되었을 수 있습니다."
                
            # 최신 기사 최대 10개 반환
            formatted_news = [f"🏅 {news}" for news in news_list[:10]]
            return "\n".join(formatted_news)
            
        except Exception as e:
            return f"네이버 스포츠 크롤링 에러 발생: {str(e)}"
        finally:
            await browser.close()

@mcp.tool()
async def get_weekly_top_news(keyword: str = None) -> str:
    """
    최근 1주일 동안 발생한 주요 뉴스를 Playwright를 통해 수집하고 요약하여 반환합니다.
    keyword 인자가 주어지면 해당 키워드와 관련된 뉴스를 검색합니다.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    query = keyword if keyword else "뉴스"
    # tbs=qdr:w 는 최근 1주일 필터
    search_url = f"https://news.google.com/search?q={query}&hl=ko&gl=KR&ceid=KR:ko&tbs=qdr:w"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(locale="ko-KR")

        try:
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            # 추가 대기: 자바스크립트 렌더링 완료
            await page.wait_for_timeout(3000)

            articles = await page.query_selector_all("article")
            results = []

            for article in articles[:15]:
                try:
                    title_el = await article.query_selector("a[href*='articles']")
                    if not title_el:
                        continue
                    title = (await title_el.inner_text()).strip()
                    rel_link = await title_el.get_attribute("href")
                    link = f"https://news.google.com{rel_link}" if rel_link else ""

                    # 본문 요약 발췌
                    snippet_el = await article.query_selector("h3 ~ div, div[data-n-tid], span[role='text']")
                    snippet = (await snippet_el.inner_text()).strip() if snippet_el else ""

                    results.append(f"- **{title}**\n  - **링크:** {link}\n  - **요약:** {snippet}")
                except:
                    continue

            if not results:
                return "뉴스를 찾을 수 없습니다. 사이트 구조가 변경되었거나 네트워크 오류일 수 있습니다."

            header = f"### 📰 최근 1주일 주요 뉴스 브리핑 (기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})"
            if keyword:
                header += f"\n\n**검색어:** {keyword}"

            return header + "\n\n" + "\n\n".join(results[:10])

        except Exception as e:
            return f"### ❌ 뉴스 크롤링 중 오류 발생\n\n**에러 내용:** {str(e)}\n\n잠시 후 다시 시도해주세요. (참고: `playwright install`이 필요할 수 있습니다.)"
        finally:
            await browser.close()

# Run the server if the script is executed directly
if __name__ == "__main__":
   print("Starting MCP server...")
   mcp.run(transport="stdio")

