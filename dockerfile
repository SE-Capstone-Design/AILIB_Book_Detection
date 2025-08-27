# Python 이미지 사용 (개발 버전 맞추기)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정 - docker 컨태이너 내에 작업 디텍토리 이름
WORKDIR /app


# OpenCV 실행에 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver


# 소스 코드 복사
COPY . .

# 컨테이너 시작 시 실행할 명령어
CMD ["uvicorn", "tracking_method.tracking_server:app", "--host", "0.0.0.0", "--port", "8000"]