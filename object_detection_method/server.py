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
from av import VideoFrame
from ultralytics import YOLO
import queue
import threading
from object_processing import ocr_with_row_clustering

# E:
# cd AILIB_OBJECTDETECTION
# uvicorn server:app --host 0.0.0.0 --port 8000 --reload

app = FastAPI()
pcs = set()
model = YOLO("../weights.pt")

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
        self.thread = threading.Thread(target=self._yolo_thread, daemon=True)
        self.thread.start()

    def _send_datachannel_safe(self, message: str):
        # 데이터채널이 열렸는지 확인 후, 이벤트루프에 안전하게 예약
        if self.data_channel and self.data_channel.readyState == "open":
            print("data_channel open")
            self.loop.call_soon_threadsafe(self.data_channel.send, message)
    def _yolo_thread(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (640, 640))

                results = model(img)[0]
                ocr_result = ocr_with_row_clustering(img, results)

                # ✅ ndarray → list 변환 후 JSON 직렬화
                try:
                    
                    message = json.dumps(ocr_result.tolist() if hasattr(ocr_result, "tolist") else ocr_result)
                    # message = {"a":"test"}
                 
                    self._send_datachannel_safe(message)
                except Exception as e:
                    print("DataChannel send error:", e)

                img_with_boxes = results.plot()
                new_frame = VideoFrame.from_ndarray(img_with_boxes, format="bgr24")
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
        if pc.connectionState in ["failed", "closed"]:
            print(f"연결 끊김: {pc.connectionState}")
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(description)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}