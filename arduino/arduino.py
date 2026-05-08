import serial
import time

# 장치 관리자에서 확인 포트로 수정하세요 (예: 'COM5', 'COM4' 등)
target_port = 'COM5'

try:
    #시리얼 포트 열기
    py_serial=serial.Serial(
        port=target_port,
        baudrate=9600,
        timeout=1
    )
    print(f"{target_port} 연결 성공!")
    while True:
        command=input("명령을 입력하세요 (1: 켜기, 0: 끄기, q: 종료)")
        if command=='1':
            py_serial.write(b'H') #아두이노로 'H' 전송
            print("LED ON")
        elif command=='0':
            py_serial.write(b'L') #아두이노로 'L' 전송
            print("LED OFF")
        elif command=='q':
            print("프로그램 종료")
            break
        else:
            print("잘못된 명령입니다.")
except Exception as e:
    print(f"연결 오류: {e}")
finally:
    if 'py_serial' in locals() and py_serial.isOpen():
        py_serial.close()
        print(f"{target_port} 연결 종료")
