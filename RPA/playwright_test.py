from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # 1. 브라우저 실행 (headless=False 로 설정하면 실제 브라우저 창이 뜨는 것을 볼 수 있음)
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 2. 웹페이지 이동
    page.goto("https://news.ycombinator.com/")

    # 3. 페이지 제목 출력 및 스크린샷 저장
    print("페이지 제목:", page.title())
    page.screenshot(path="hn_screenshot.png")

    # 4. 브라우저 종료
    browser.close()
