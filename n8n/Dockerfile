FROM n8nio/n8n:latest

USER root

# 데비안 기반 공식 이미지에 맞게 apt로 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# uv 전역 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

USER node