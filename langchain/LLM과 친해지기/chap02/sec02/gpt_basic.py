from openai import OpenAI

api_key = 'sk-**************'	# ① 

client = OpenAI(api_key=api_key)

# ② 
response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.1,  # ③
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "2022년 월드컵 우승팀은 어디야?"},
  ]		# ④
)

print(response)

print('----')	# ⑤
print(response.choices[0].message.content) 
