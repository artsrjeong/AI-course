import serial
import time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP()

# 서버 시작 시 포트를 한 번만 열고 연결을 유지합니다.
TARGET_PORT = 'COM4'
BAUD_RATE = 9600

try:
    py_serial = serial.Serial(port=TARGET_PORT, baudrate=BAUD_RATE, timeout=1)
    # 아두이노가 리셋 후 대기 상태가 될 때까지 2초간 대기합니다.
    time.sleep(2)
except Exception as e:
    print(f"시리얼 포트 연결 실패: {e}")
    py_serial = None

@mcp.tool()
def arduino_led_control(command: str) -> str:
    """
    Control LED of Arduino based on command ('1' = ON, '0' = OFF)
    """
    if py_serial is None or not py_serial.is_open:
        return "오류: 아두이노 시리얼 포트가 연결되어 있지 않습니다."

    try:
        if command == '1':
            py_serial.write(b'H')
            return "LED ON"
        elif command == '0':
            py_serial.write(b'L')
            return "LED OFF"
        else:
            return "Invalid command"
    except Exception as e:
        return f"전송 오류: {e}"

if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run(transport="stdio")
