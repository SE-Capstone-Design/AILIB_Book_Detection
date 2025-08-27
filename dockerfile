# Python 이미지 사용 (개발 버전 맞추기)
FROM pytorch/pytorch:2.7-cuda12.8-cudnn8-runtime

# 작업 디렉토리 설정 - docker 컨태이너 내에 작업 디텍토리 이름
WORKDIR /app


# 의존성 파일 복사 
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# 소스 코드 복사
COPY . .

# 컨테이너 시작 시 실행할 명령어
CMD ["uvicorn", "tracking_method.tracking_server:app", "--host", "0.0.0.0", "--port", "8000"]