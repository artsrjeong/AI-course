
import json
import requests

url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

# 사용자 토큰
headers = {
  "Content-Type": "application/x-www-form-urlencoded",
  "Authorization": "Bearer " + \
  "ga46iHLr4FI0cnKeHMjl8n-a8YbkCKAUAAAAAQoXEpYAAAGe-DQg19Q0RDl69jWm"
}

data = {
  "template_object" : json.dumps({ 
  "object_type" : "text",
  "text" : "침입이 감지 되었습니다!!",
  "link" : {
    "web_url" : "http://www.jkelec.co.kr"
  },
 })
}

response = requests.post(url, headers=headers, data=data)
print(response.status_code)
if response.json().get('result_code') == 0:
    print('Message send successed.')
else:
    print('Message send failed. : ' + str(response.json()))
