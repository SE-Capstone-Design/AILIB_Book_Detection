# Python 이미지 사용
FROM Python:3.8.20

# 작업 디렉토리 설정 - docker 컨태이너 내에 작업 디텍토리 이름
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 컨테이너 시작 시 실행할 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]