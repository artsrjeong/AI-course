from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
# ② Client 객체 생성
client = genai.Client(api_key=api_key)

# ③ 및 ④ 모델 호출 및 설정
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",  # Gemini 기본 제공 모델
    contents="2022년 월드컵 우승팀은 어디야?",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",  # system 역할 지정
        temperature=0.1,  # 온도(창의성/확률) 설정
    )
)

print(response)

print('----') # ⑤
print(response.text) # 답변 내용 출력