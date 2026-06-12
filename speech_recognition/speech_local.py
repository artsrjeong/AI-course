import speech_recognition as sr
from gtts import gTTS
from audioplayer import AudioPlayer
import os
from dotenv import load_dotenv
import time

load_dotenv()

def speak(text):
    """텍스트를 음성으로 바꾸어 스피커로 출력하는 함수"""
    tts = gTTS(text=text, lang='ko')
    filename = 'voice.mp3'
    tts.save(filename)
    AudioPlayer(filename).play(block=True)
    os.remove(filename)

def simple_respond(user_input):
    """간단한 키워드 기반 응답 (로컬, API 없음)"""
    responses = {
        "안녕": "안녕하세요! 반갑습니다.",
        "안녕하세요": "안녕하세요! 뭘 도와드릴까요?",
        "이름": "저는 음성 어시스턴트입니다.",
        "시간": "지금 시간을 알려드릴게요.",
        "날씨": "날씨 정보는 따로 확인이 필요합니다.",
        "감사": "도움이 되어서 기쁩니다!",
        "고마워": "별말씀을요!",
    }
    
    # 입력된 텍스트에 포함된 키워드 찾기
    for keyword, response in responses.items():
        if keyword in user_input:
            return response
    
    return "잠깐만요, 이해가 안 됩니다. 다시 말씀해 주실 수 있나요?"

def listen_and_respond():
    r = sr.Recognizer()
    
    # 사용 가능한 입력 마이크: [2] 웹캠, [4] USB 마이크, [34] USB2.0 Device 등
    microphone_index = 3
    
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
        
        # 로컬 키워드 기반 응답 (API 사용 안 함)
        answer = simple_respond(user_input)
        print(f"🤖 어시스턴트: {answer}")
        
        # 답변을 음성으로 출력
        speak(answer)
        
    except sr.UnknownValueError:
        print("❌ 음성을 이해하지 못했습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    listen_and_respond()
