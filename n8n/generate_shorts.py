import sys
import json
import os
import requests
from gtts import gTTS  # 구글 TTS 사용

# 윈도우 한글 인코딩 깨짐 방지
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Pixabay API Key 설정 (본인의 키 입력)
PIXABAY_API_KEY = "58000181-56d7c060e1abab72e931c6837"

def download_pixabay_image(query, output_path):
    """Pixabay API를 이용해 안전한 키워드로 이미지를 검색하고 다운로드"""
    # 씬 번호나 순서에 따라 무난한 배경 키워드 매칭
    fallback_keywords = ["aesthetic background", "dark mood background", "cinematic landscape", "abstract texture"]
    search_query = "aesthetic background" # 기본값
    
    # query가 너무 길거나 복잡하면 Pixabay가 에러를 내므로 단순 키워드 추출 시도
    if query:
        # 영어 포함 여부 확인 간단 체크 후, 너무 길면 기본 키워드로 대체
        if len(query) < 30 and all(ord(c) < 128 for c in query):
            search_query = query
            
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={requests.utils.quote(search_query)}&image_type=photo&orientation=vertical&safesearch=true&per_page=3"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get("hits") and len(data["hits"]) > 0:
            image_url = data["hits"][0]["largeImageURL"]
            img_data = requests.get(image_url).content
            with open(output_path, "wb") as f:
                f.write(img_data)
            print(f"이미지 다운로드 성공: {output_path} (검색어: {search_query})")
        else:
            # 검색 결과가 없으면 첫 번째 폴백 키워드로 재시도
            fallback_url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q=background&image_type=photo&orientation=vertical&safesearch=true&per_page=3"
            res2 = requests.get(fallback_url)
            data2 = res2.json()
            if data2.get("hits"):
                img_data = requests.get(data2["hits"][0]["largeImageURL"]).content
                with open(output_path, "wb") as f:
                    f.write(img_data)
                print(f"이미지 다운로드 성공 (대체 검색): {output_path}")
            else:
                print(f"경고: 이미지를 찾지 못했습니다.")
    except Exception as e:
        print(f"에러: 이미지 다운로드 중 오류 발생 ({e})")
        
def main():
    # 입력 파일 경로 설정
    input_json_path = r"E:\Project\AI-course\n8n\output\input_data.json"
    
    if not os.path.exists(input_json_path):
        print(f"에러: 입력 JSON 파일을 찾을 수 없습니다: {input_json_path}")
        sys.exit(1)
        
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"에러: JSON 파일을 파싱하는 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    title = data.get("title", "쇼츠_영상")
    scenes = data.get("scenes", [])
    
    print(f"=== [쇼츠 자동화 & 에셋 다운로드 시작] ===")
    print(f"제목: {title}")
    print(f"총 씬 수: {len(scenes)}개\n")

    output_dir = r"E:\Project\AI-course\n8n\output\assets"
    os.makedirs(output_dir, exist_ok=True)

    for i, scene in enumerate(scenes):
        scene_num = scene.get("scene_number", i + 1)
        narration = scene.get("narration", "")
        image_prompt = scene.get("image_prompt", "")

        print(f"--- [Scene {scene_num} 처리 중] ---")

        # 1. gTTS 오디오 생성 (.mp3)
        audio_path = os.path.join(output_dir, f"scene_{scene_num}.mp3")
        print(f"음성 생성 중: {narration[:20]}...")
        tts = gTTS(text=narration, lang='ko', slow=False)
        tts.save(audio_path)
        print(f"오디오 저장 완료: {audio_path}")

        # 2. Pixabay 이미지 다운로드 (.jpg)
        image_path = os.path.join(output_dir, f"scene_{scene_num}.jpg")
        search_keyword = image_prompt if image_prompt else "aesthetic background"
        download_pixabay_image(search_keyword, image_path)
        print("-" * 30)

    # 최종 합성용 데이터 경로 저장
    current_shorts_path = r"E:\Project\AI-course\n8n\output\current_shorts.json"
    with open(current_shorts_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n성공: 모든 에셋 준비 및 JSON 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()