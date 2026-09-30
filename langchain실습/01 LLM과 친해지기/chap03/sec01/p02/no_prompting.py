import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="오리",
    config=types.GenerateContentConfig(
        system_instruction="너는 유치원 학생이야. 유치원생처럼 답변해줘.",
        temperature=0.9,
    )
)
print(response)
print('----')	# ⑤
print(response.text) 
