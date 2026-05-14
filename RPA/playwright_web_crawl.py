from playwright.sync_api import sync_playwright
import time

def main():
    url = "https://www.du.ac.kr/main.do"
    
    with sync_playwright() as p:
        # 브라우저 실행 (headless=False로 설정하여 동작 확인 가능)
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            page.goto(url)

            # 요소가 나타날 때까지 대기 및 클릭 (XPATH 사용)
            page.click("//*[@id='container']/div[3]/div/div[1]/ul/li[1]/button")
            
            # 뉴스 목록 가져오기 (CSS Selector 사용)
            # selenium_web_crawl.py의 로직을 따라 CSS selector로 여러 요소를 찾습니다.
            articles = page.query_selector_all("#container > div.main-contents02 > div")
            
            for article in articles:
                title = article.inner_text().strip()
                if title:
                    print(f"뉴스: {title}")
                    
        except Exception as e:
            print(f"에러 발생: {e}")
            
        finally:
            # 확인을 위해 잠시 대기 후 종료
            time.sleep(5)
            browser.close()

if __name__ == "__main__":
    main()
