import speech_recognition as sr

try:
    mics = sr.Microphone.list_microphone_names()
    print("=== 인식된 마이크 목록 ===")
    for i, name in enumerate(mics):
        print(f"[{i}]: {name}")
    if not mics:
        print("인식된 오디오 입력 장치가 전혀 없습니다.")
    else:
        print(f"\n기본 마이크: [{sr.Microphone().device_index}]")
except Exception as e:
    print(f"에러 발생: {e}")