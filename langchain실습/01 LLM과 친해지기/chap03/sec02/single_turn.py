import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # 환경 변수에서 GEMINI_API_KEY 가져오기

client = genai.Client(api_key=api_key)  # Gemini 클라이언트 인스턴스 생성

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="너는 사용자를 도와주는 상담사야.",
            temperature=0.9,
        ),
        contents=user_input,
    )
    print("AI: " + response.text)