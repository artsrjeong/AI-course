#import spidev
import time
import pymysql    # pymysql 임포트

# def analog_read(channel):
#     r = spi.xfer2([1, (0x08+channel)<<4, 0])
#     adc_out = ((r[1]&0x03)<<8) + r[2]
#     return adc_out

# spi = spidev.SpiDev()
# spi.open(0,0)
# spi.max_speed_hz = 1000000

# 전역변수 선언부 
db = None 
cur = None 

# 접속정보
db = pymysql.connect(host='172.26.227.149', user='root', password='jung7574', db='mysql', charset='utf8')

try:
  cur = db.cursor() # 커서생성 
  
  while True:
    # adc = analog_read(3)
    # voltage = adc*(3.3/1023/5)*1000
    voltage=250
    temperature = voltage / 10.0
    #print ("%4d/1023 => %5.3f V => %4.1f°C" % (adc, voltage, temperature))  
	
    sql = "INSERT INTO temperature(TEMP) VALUES (%4.1f)" %  temperature
    print(sql)

    # 실행할 sql문 
    cur.execute(sql)
	
    # 커서로 sql문 실행
    db.commit() # 저장 
	
    time.sleep(10)
except KeyboardInterrupt:
  pass	
# Finally:
#   spi.close()
