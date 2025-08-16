from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO 모델 로드
model = YOLO(r"C:\Users\user\Downloads\runs\detect\train\weights\best.pt")
if torch.cuda.is_available():
    model.to('cuda')

@app.websocket("/ws/video")
async def websocket_yolo(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            # 프론트에서 전송한 바이트 이미지 수신
            byte_data = await websocket.receive_bytes()
            np_arr = np.frombuffer(byte_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # YOLO 추론
            results = model(frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            labels = results.boxes.cls.cpu().numpy().astype(int)
            names = results.names if hasattr(results, "names") else model.names
            print(results)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                label = names[labels[i]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 결과 이미지를 바이트로 인코딩 후 전송
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())

        except Exception as e:
            print("WebSocket Error:", e)
            break
