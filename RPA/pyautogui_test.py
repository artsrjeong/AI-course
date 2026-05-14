import pyautogui
import pyperclip
import time

# while True:
#     print(pyautogui.position())
#     time.sleep(0.1)

# pyautogui.moveTo(1215,455,duration=1)
# pyautogui.click()
# pyautogui.write('python logo', interval=0.1)
# pyautogui.press('enter')

#한글 입력

pyautogui.moveTo(1215,455,duration=1)
pyautogui.click()
time.sleep(0.5)
pyperclip.copy("파이썬 로고")
pyautogui.hotkey("Ctrl","v")
time.sleep(0.5)
pyautogui.press("enter")