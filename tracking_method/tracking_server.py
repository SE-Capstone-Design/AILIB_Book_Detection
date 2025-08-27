import cv2
import numpy as np
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from aiortc import (
    RTCPeerConnection, MediaStreamTrack, RTCSessionDescription,
    RTCDataChannel
)
import torch

from av import VideoFrame
from ultralytics import YOLO
import queue
import threading
from tracking_method.state import ManageItem 
from tracking_method.processing import row_ocr_clustering
from tracking_method.boundingBox import draw_bounding_box

import os

# === NEW: supervision (ByteTrack + Annotators) ===
import supervision as sv

app = FastAPI()
pcs = set()
base_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_dir, "weights.pt")
model = YOLO(weights_path)  # 탐지 모델 그대로 사용

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class YoloTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, data_channel=None, loop=None):
        super().__init__()
        self.track = track
        self.data_channel = data_channel
        self.loop = loop or asyncio.get_event_loop()
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.manage = ManageItem() 
        print(torch.cuda.is_available()) 
        print(torch.version.cuda)  
        print(torch.cuda.get_device_name(0))

        # === NEW: ByteTrack & Annotators ===
        self.tracker = sv.ByteTrack()  
        self.thread = threading.Thread(target=self._yolo_thread, daemon=True)
        self.thread.start()

    def _send_datachannel_safe(self, message: str):
        if self.data_channel and self.data_channel.readyState == "open":
            self.loop.call_soon_threadsafe(self.data_channel.send, message)

    def _yolo_thread(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                img_orignal = frame.to_ndarray(format="bgr24")
                # YOLO 입력 크기 통일 (선택)
                img = cv2.resize(img_orignal, (640, 640))

                # 1) 탐지
                results = model(img, verbose=False)[0]

                # 2) 탐지 → Detections 변환
                detections = sv.Detections.from_ultralytics(results)
                # (필요 시 감도 조정) conf 필터 예: detections = detections[detections.confidence > 0.25]

                # 3) 트래킹 업데이트 (ByteTrack)
                tracked = self.tracker.update_with_detections(detections)

              
                try:
                    # 4) 시각화 (ID 라벨)
                    r_o_c = row_ocr_clustering(img,tracked,img_orignal)
                    if len(r_o_c) > 0:
                        total = self.manage.start(r_o_c)
                        draw_bounding_box(img, total)                                    
                        # 5) 결과 datachnannel 전송    
                        self._send_datachannel_safe(json.dumps(total))             
                                                
                except Exception as e:
                    print("DataChannel send error:", e)

                # 6) 결과 프레임 교체
                new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                with self.lock:
                    self.result_frame = new_frame

            except queue.Empty:
                continue

    async def recv(self):
        frame = await self.track.recv()

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        with self.lock:
            return self.result_frame if self.result_frame else frame

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.get("/offer")
async def offer(request: Request):
    params = await request.json()
    description = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    loop = asyncio.get_event_loop()
    data_channel_holder = {"ch": None}
    yolo_track_holder = {"track": None}


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    description = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    loop = asyncio.get_event_loop()
    data_channel_holder = {"ch": None}
    yolo_track_holder = {"track": None}

    @pc.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        print("📨 server got datachannel:", channel.label, channel.protocol)
        data_channel_holder["ch"] = channel
        # YOLO 트랙이 이미 만들어졌다면 연결
        if yolo_track_holder["track"] is not None:
            yolo_track_holder["track"].data_channel = channel  
    
        @channel.on("close")
        def on_close():
            print(f"DataChannel closed: {channel.label}")        
                    
            
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            yolo_track = YoloTrack(track, data_channel=data_channel_holder["ch"], loop=loop)
            yolo_track_holder["track"] = yolo_track
            pc.addTrack(yolo_track)
            
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("connectionstatechange")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(description)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}