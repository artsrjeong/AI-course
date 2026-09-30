import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
# .env 파일에서 GEMINI_API_KEY를 불러옵니다.
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

# ② Gemini 모델 호출
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    config=types.GenerateContentConfig(
        system_instruction="너는 유치원 학생이야. 유치원생처럼 답변해줘.",
        temperature=0.9,  # ③
    ),
    contents=[
        {"role": "user", "parts": [{"text": "참새"}]},
        {"role": "model", "parts": [{"text": "짹짹"}]},
        {"role": "user", "parts": [{"text": "오리"}]},
    ],  # ④
)

print(response)

print('----')  # ⑤
print(response.text)