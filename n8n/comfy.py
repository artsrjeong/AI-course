import json
import time
import requests

def download_comfy_image(query, output_path):
  """ComfyUI API를 이용해 쿼리 기반 이미지를 생성하고 output_path에 저장합니다."""
    
  # ComfyUI 로컬 서버 주소
  COMFY_URL = "http://127.0.0.1:8188"
  workflow_api_json_path = r"E:\Project\AI-course\n8n\workflow_api.json"
  # 1. 사전에 저장해둔 API 형식의 워크플로우 JSON 불러오기
  try:
    with open(workflow_api_json_path, "r", encoding="utf-8") as f:
      workflow = json.load(f)
  except FileNotFoundError:
    raise Exception(
        "workflow_api.json 파일이 없습니다. ComfyUI에서 'Save (API Format)'으로"
        " 추출 후 스크립트 경로에 넣어주세요."
    )

  # 2. 텍스트 프롬프트 노드를 찾아 사용자가 전달한 query(프롬프트)로 교체
  # (ComfyUI 워크플로우에서 CLIPTextEncode 노드의 "text" 입력을 동적으로 변경)
  prompt_updated = False
  for node_id, node_data in workflow.items():
    if (
        node_data.get("class_type") == "CLIPTextEncode"
        and "text" in node_data["inputs"]
    ):
      # 긍정 프롬프트(Positive Prompt) 영역에 query 주입
      node_data["inputs"]["text"] = query
      prompt_updated = True
      break

  if not prompt_updated:
    raise Exception(
        "워크플로우에서 CLIPTextEncode 노드를 찾지 못했습니다. API JSON"
        " 설정을 확인해주세요."
    )

  # 3. ComfyUI 서버에 이미지 생성 작업 큐(Queue) 등록 요청
  payload = {"prompt": workflow}
  response = requests.post(f"{COMFY_URL}/prompt", json=payload)
  resp_data = response.json()
  prompt_id = resp_data.get("prompt_id")

  if not prompt_id:
    raise Exception(f"ComfyUI 작업 등록 실패: {resp_data}")

  print(
      f"🎨 [ComfyUI] 이미지 생성 작업 전송됨 (Prompt ID: {prompt_id}). 생성"
      " 대기 중..."
  )

  # 4. 작업이 완료될 때까지 상태 확인 (Polling Loop)
  image_filename = None
  subfolder = ""
  folder_type = "output"

  while True:
    history_res = requests.get(f"{COMFY_URL}/history/{prompt_id}")
    history_data = history_res.json()

    if prompt_id in history_data:
      # 작업 완료 시점의 출력 결과물 정보 추출
      outputs = history_data[prompt_id]["outputs"]
      for node_output in outputs.values():
        if "images" in node_output:
          image_info = node_output["images"][0]
          image_filename = image_info["filename"]
          subfolder = image_info["subfolder"]
          folder_type = image_info["type"]
          break
      break
    time.sleep(1)  # 1초마다 완료 여부 체크

  if not image_filename:
    raise Exception("이미지 생성 결과 파일을 찾지 못했습니다.")

  # 5. 생성된 이미지 파일 다운로드 및 지정된 경로에 저장
  image_url_params = {
      "filename": image_filename,
      "subfolder": subfolder,
      "type": folder_type,
  }
  img_res = requests.get(f"{COMFY_URL}/view", params=image_url_params)

  with open(output_path, "wb") as f:
    f.write(img_res.content)

  print(f"✨ 이미지 다운로드 및 저장 완료: {output_path}")


# --- 실행 테스트 예시 ---
if __name__ == "__main__":
  test_query = (
      "Cinematic vertical 9:16 background photo of an hourglass with sand"
      " quickly slipping away, dark moody atmosphere, realistic lighting, 8k"
      " resolution"
  )
  download_comfy_image(test_query, "hourglass_output.png")