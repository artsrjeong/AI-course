from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def main():
    url = "https://www.du.ac.kr/main.do"
    
    # Selenium 4.6+ automatically manages the driver
    driver = webdriver.Edge()
    
    try:
        driver.get(url)

        # 요소가 로딩될 때까지 최대 10초 대기
        wait = WebDriverWait(driver, 10)
        
        driver.find_element(By.XPATH, "//*[@id='container']/div[3]/div/div[1]/ul/li[1]/button").click()
        articles = driver.find_elements(By.CSS_SELECTOR, "#container > div.main-contents02 > div")
        for article in articles:
            title = article.text.strip()
            if title:
                print(f"뉴스: {title}")
                
    except Exception as e:
        print(f"에러 발생: {e}")
        
    finally:
        # 확인을 위해 잠시 대기 후 종료
        time.sleep(5)
        driver.quit()

if __name__ == "__main__":
    main()