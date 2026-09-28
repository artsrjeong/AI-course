import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# ② Client 객체 생성
client = genai.Client(api_key=api_key)

# ③ 및 ④ 모델 호출 및 설정
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="세상에서 누가 제일 아름답니?",
    config=types.GenerateContentConfig(
        system_instruction="너는 백설공주 이야기 속의 거울이야. 그 이야기 속의 마법 거울의 캐릭터에 부합하게 답변해줘.",
        temperature=0.9,
    )
)

print(response)

print('----') # ⑤
print(response.text)