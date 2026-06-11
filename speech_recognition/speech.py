import speech_recognition as sr
from google import genai
from gtts import gTTS
from audioplayer import AudioPlayer
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

# 1. Gemini 클라이언트 설정 (API 키 필요)
client = genai.Client(api_key=GOOGLE_API_KEY)

def speak(text):
    """텍스트를 음성으로 바꾸어 스피커로 출력하는 함수"""
    tts = gTTS(text=text, lang='ko')
    filename = 'voice.mp3'
    tts.save(filename)
    AudioPlayer(filename).play(block=True)
    os.remove(filename)

def listen_and_respond():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("듣고 있어요... 말씀하세요.")
        audio = r.listen(source)
        
    try:
        # 구글 STT로 음성 -> 텍스트 변환
        user_input = r.recognize_google(audio, language='ko-KR')
        print(f"사용자: {user_input}")
        
        # Gemini API로 답변 생성
        response = client.models.generate_content(
            model='gemini-2.5-flash', # 혹은 gemini-1.5-flash
            contents=user_input
        )
        answer = response.text
        print(f"Gemini: {answer}")
        
        # 답변을 음성으로 출력
        speak(answer)
        
    except sr.UnknownValueError:
        print("음성을 이해하지 못했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    listen_and_respond()