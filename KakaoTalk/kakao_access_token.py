import requests
# 인가 코드는 https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=YOUR_REST_API_KEY&redirect_uri=https://localhost:3000&scope=talk_message 
# 위에 접속해서 REST_API_KEY 만 변경하고 인증하면 새로운 web 페이지 주소창에 code가 나옴


# 1. 본인의 정보로 변경하세요
rest_api_key = "06741192c11444ac734cfd757690fa81"  # 아까 확인하신 REST API 키
redirect_uri = "https://localhost:3000"
authorize_code = "SXu9AHFe7cVbjxDFn-3yBM2AHfv9zNL7dPpWgTrFC9ms_aoGqL1KsAAAAAQKFwYuAAABnvc5ffbC3p98Pd5TpQ" 

url = "https://kauth.kakao.com/oauth/token"
data = {
    "grant_type": "authorization_code",
    "client_id": rest_api_key,
    "redirect_uri": redirect_uri,
    "code": authorize_code,
}

response = requests.post(url, data=data)
tokens = response.json()

# 2. 결과 확인
if "access_token" in tokens:
    print("\n[발급 성공] 아래 토큰을 저장해서 메시지 보낼 때 사용하세요:")
    print(tokens["access_token"])
else:
    print("\n[발급 실패] 에러 메시지:", tokens)