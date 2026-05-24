from mcp.server.fastmcp import FastMCP # Parameter is not strictly needed now, but good practice to keep if you add more complex params later
import asyncio
from playwright.async_api import async_playwright

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

# Run the server if the script is executed directly
if __name__ == "__main__":
   print("Starting MCP server...")
   mcp.run(transport="stdio")

