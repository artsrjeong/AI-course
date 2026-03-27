import serial
import time

target_port = "COM3"
#장치관리자에서 확인한 포트 번호로 수정하세요(예: 'COM3','COM4' 등)
try:
    #시리얼 포트 열기
    py_serial=serial.Serial(
        port=target_port,
        baudrate=9600,
        timeout=1
    )
    print(f"{target_port} 연결 성공!")
    while True:
        command = input("명령어 입력하세요: (1: LED 켜기, 2: LED 끄기, q: 프로그램 종료)")
        if command == 'q':
            break
        elif command == '1':
            py_serial.write(b'H') #아두이노로 'H' 전송
            print("LED 켜짐")
        elif command == '0':
            py_serial.write(b'L') #아두이노로 'L' 전송
            print("LED 꺼짐")
        else:
            print("잘못된 명령어입니다.")   
    py_serial.close() #시리얼 포트 닫기
except Exception as e:
    print(f"오류 발생: {e}")
