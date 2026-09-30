import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

# ①
def get_ai_response(contents):
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="너는 사용자를 도와주는 상담사야.",
            temperature=0.9,
        ),
        contents=contents,  # 대화 기록 전달
    )
    return response.text

# 대화 기록 리스트 초기화
contents = []

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break
    
    # 사용자 메시지 추가 (Gemini 역할: user)
    contents.append({"role": "user", "parts": [{"text": user_input}]})
    
    ai_response = get_ai_response(contents)
    
    # AI 메시지 추가 (Gemini 역할: model)
    contents.append({"role": "model", "parts": [{"text": ai_response}]})

    print("AI: " + ai_response)