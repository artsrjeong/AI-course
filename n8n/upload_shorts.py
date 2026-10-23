import os
import sys
import argparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 윈도우 한글 인코딩 깨짐 방지
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# YouTube API 범위 설정 (영상 업로드 권한)
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

CLIENT_SECRETS_FILE = 'client_secrets.json'
TOKEN_FILE = 'token.json'

def get_authenticated_service():
    """OAuth 2.0 인증을 진행하고 YouTube API 객체를 반환합니다."""
    creds = None
    # 기존에 저장된 토큰이 있는지 확인
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    # 토큰이 없거나 유효하지 않으면 재인증 진행
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                print(f"Error: '{CLIENT_SECRETS_FILE}' 파일이 없습니다. Google Cloud Console에서 다운로드하여 배치하세요.")
                sys.exit(1)
            
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # 다음 실행을 위해 토큰을 token.json으로 저장
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)

def upload_video(file_path, title, description, category_id="22", privacy_status="private"):
    """비디오를 유튜브에 업로드합니다."""
    if not os.path.exists(file_path):
        print(f"Error: 업로드할 파일이 존재하지 않습니다: {file_path}")
        sys.exit(1)

    youtube = get_authenticated_service()

    # 메타데이터 구성 (제목에 #Shorts가 포함되어야 쇼츠로 인식됨)
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'categoryId': category_id  # 22: People & Blogs (기본값)
        },
        'status': {
            'privacyStatus': privacy_status,  # 'private', 'public', 'unlisted' 중 선택
            'selfDeclaredMadeForKids': False   # 아동용 여부
        }
    }

    # 대용량 파일 분할 업로드 설정 (chunksize=-1: 한 번에 업로드)
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True, mimetype='video/mp4')

    request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )

    print(f"[{title}] 업로드를 시작합니다...")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"업로드 진행률: {int(status.progress() * 100)}%")

    print("\n✅ 업로드 완료!")
    print(f"비디오 ID: {response.get('id')}")
    print(f"유튜브 링크: https://youtu.be/{response.get('id')}")
    return response.get('id')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YouTube Shorts 자동 업로드 스크립트")
    parser.add_argument('--file', required=True, default="final_shorts.mp4",help="업로드할 MP4 파일 경로")
    parser.add_argument('--title', required=True, default="YouTube Shorts", help="유튜브 제목 (#Shorts 포함 권장)")
    parser.add_argument('--description', default="", help="유튜브 영상 설명")
    parser.add_argument('--privacy', default="private", choices=["private", "public", "unlisted"], 
                        help="공개 범위 (private: 비공개, unlisted: 일부공개, public: 공개)")

    args = parser.parse_args()

    upload_video(
        file_path=args.file,
        title=args.title,
        description=args.description,
        privacy_status=args.privacy
    )