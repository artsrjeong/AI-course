void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); //파이썬과 맞춘 통신 속도
  pinMode(13,OUTPUT); // 아두이노 내장 LED (13번 핀)
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0){
    char cmd=Serial.read();  // 파이썬이 보낸 글자 읽기
    if(cmd=='H'){
      digitalWrite(13,HIGH); //LED 켜기
    } else if(cmd=='L'){
      digitalWrite(13,LOW);
    }
  }
}
