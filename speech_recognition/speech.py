import speech_recognition as sr
from google import genai
from gtts import gTTS
from audioplayer import AudioPlayer
import os
from dotenv import load_dotenv
import time

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# API 키 검증
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 1. Gemini 클라이언트 설정 (API 키 필요)
client = genai.Client(api_key=GEMINI_API_KEY)

def speak(text):
    """텍스트를 음성으로 바꾸어 스피커로 출력하는 함수"""
    tts = gTTS(text=text, lang='ko')
    filename = 'voice.mp3'
    tts.save(filename)
    AudioPlayer(filename).play(block=True)
    os.remove(filename)

def listen_and_respond():
    r = sr.Recognizer()
    r.energy_threshold = 4000  # 마이크 감도 조정
    
    # 사용 가능한 입력 마이크: [2] 웹캠, [4] USB 마이크, [34] USB2.0 Device 등
    microphone_index = 3  # 기본값: 웹캠 마이크 (2 또는 4로 변경 가능)
    
    try:
        # 마이크 접근
        with sr.Microphone(device_index=microphone_index) as source:
            # 환경 소음에 따라 자동으로 감도 조정 (1초)
            r.adjust_for_ambient_noise(source, duration=1)
            print("🎤 마이크에서 듣고 있어요... 말씀하세요.")
            audio = r.listen(source, timeout=10)  # 10초 타임아웃
            
    except sr.RequestError:
        print(f"❌ 마이크 인덱스 {microphone_index}에 접근할 수 없습니다.")
        print("check_device.py를 실행해서 올바른 마이크 인덱스를 확인하세요.")
        return
    except sr.UnknownValueError:
        print("❌ 음성을 감지할 수 없습니다. 마이크 설정을 확인하세요.")
        return
    except Exception as e:
        print(f"❌ 마이크 오류: {e}")
        return
        
    try:
        # 구글 STT로 음성 -> 텍스트 변환
        print("🔄 음성을 텍스트로 변환 중...")
        user_input = r.recognize_google(audio, language='ko-KR')
        print(f"👤 사용자: {user_input}")
        
        # Gemini API로 답변 생성 (재시도 로직)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash-lite',  # 더 저렴하고 빠른 모델
                    contents=user_input
                )   
                answer = response.text
                print(f"🤖 Gemini: {answer}")
                
                # 답변을 음성으로 출력
                speak(answer)
                break
                
            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    print(f"⏳ 서버 과부하... {3 - attempt}초 후 재시도합니다.")
                    time.sleep(3)
                else:
                    raise e
        
    except sr.UnknownValueError:
        print("❌ 음성을 이해하지 못했습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    listen_and_respond()