from selenium import webdriver
from selenium.webdriver.common.by import By

import time

def main():
    url="http://www.naver.com"
    # Selenium 4.6+ automatically manages the driver
    driver = webdriver.Edge()
    driver.get(url)
    time.sleep(2)
    login = driver.find_element(By.XPATH, "//*[@id='account']/div/a").click()
    time.sleep(2)
    id = driver.find_element(By.XPATH, "//*[@id='id']").send_keys("artsrjeong")
    time.sleep(2)
    driver.find_element(By.XPATH, "//*[@id='pw']").send_keys("abcd9596!@")
    time.sleep(2)
    driver.find_element(By.XPATH, "//*[@id='log.login']").click()
    time.sleep(5)
    driver.quit()
if __name__=="__main__":
    main()