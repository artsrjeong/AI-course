from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=api_key)

print("--- 사용 가능한 모델 목록 ---")
for m in client.models.list():
    print(m.name)