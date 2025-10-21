# Python 이미지 사용 (개발 버전 맞추기)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정 - docker 컨태이너 내에 작업 디텍토리 이름
WORKDIR /app


# OpenCV 실행에 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*


s#test
RUN apt-get update && apt-get install -y build-essential cmake git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_opencv_python3=ON \
          -D PYTHON_EXECUTABLE=$(which python3) .. && \
    make -j$(nproc) && make install && ldconfig 

# 의존성 파일 복사 
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver


# 소스 코드 복사
COPY . .

# 컨테이너 시작 시 실행할 명령어
CMD ["uvicorn", "tracking_method.tracking_server:app", "--host", "0.0.0.0", "--port", "8000"]