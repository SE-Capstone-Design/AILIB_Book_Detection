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
from tracking_method.processing import ocr_with_row_clustering, duplicate_text

# === NEW: supervision (ByteTrack + Annotators) ===
import supervision as sv

app = FastAPI()
pcs = set()
model = YOLO("./yolov8n.pt")  # 탐지 모델 그대로 사용

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

        # === NEW: ByteTrack & Annotators ===
        self.tracker = sv.ByteTrack()  
        self.box_annotator = sv.BoxAnnotator(thickness=2) # 경계 상자 선의 두께.
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5) #텍스트 두께.

        self.thread = threading.Thread(target=self._yolo_thread, daemon=True)
        self.thread.start()

    def _send_datachannel_safe(self, message: str):
        if self.data_channel and self.data_channel.readyState == "open":
            self.loop.call_soon_threadsafe(self.data_channel.send, message)

    def _yolo_thread(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                img = frame.to_ndarray(format="bgr24")
                # YOLO 입력 크기 통일 (선택)
                img = cv2.resize(img, (640, 640))

                # 1) 탐지
                results = model(img, verbose=False)[0]

                # 2) 탐지 → Detections 변환
                detections = sv.Detections.from_ultralytics(results)
                # (필요 시 감도 조정) conf 필터 예: detections = detections[detections.confidence > 0.25]

                # 3) 트래킹 업데이트 (ByteTrack)
                #   supervision 0.19+ 버전은 인자 없이도 동작.
                #   특정 해상도/프레임 속도 기반 튜닝이 필요하면 ByteTrackArgs로 세부설정 가능.
                tracked = self.tracker.update_with_detections(detections)

                # 4) 시각화 (ID 라벨)
                #   tracked.xyxy: (N,4), tracked.class_id, tracked.tracker_id 등 사용 가능
                #   라벨 문자열 구성
                labels = []
                for i in range(len(tracked)):
                    cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                    tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    labels.append(f"id:{tid} cls:{cls_id} conf:{conf:.2f}")

                # 박스/라벨 그리기
                img_drawn = self.box_annotator.annotate(
                    scene=img.copy(),
                    detections=tracked
                )
                img_drawn = self.label_annotator.annotate(
                    scene=img_drawn,
                    detections=tracked,
                    labels=labels
                )

                # 5)  OCR에 트래킹 정보 반영
              
                try:
                    # 예시: 트래킹 결과를 간단 JSON으로 전송
                    # payload = []
                    # for i in range(len(tracked)):
                    #     x1, y1, x2, y2 = map(float, tracked.xyxy[i])
                    #     tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    #     cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                    #     conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    #     payload.append({"id": tid, "cls": cls_id, "conf": conf, "bbox": [x1, y1, x2, y2]})
                    re =  ocr_with_row_clustering(img,detections)
                    dup =  duplicate_text(re)  
                    
                    for i in dup:
                       cv2.rectangle(img_drawn, (i[4],i[5]),(i[6],i[7]), color=(0,0,255), thickness =2)
                    #    cv2.putText(img_drawn, "diff_book", (i[4], i[5]-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (255, 255, 255), thickness =3)
                     
                                   
                    if re is not None or len(re) != 0:
                        t = []
                        for row in re:
                            t.append({
                                "row": int(row[0]),             # 행 번호
                                "text": str(row[1]),             # 텍스트
                                "id": int(row[2]),               # 인덱스
                                "bbox": [float(row[3]), float(row[4]), float(row[5]), float(row[6])] }) # 바운딩박스                        
                        self._send_datachannel_safe(json.dumps({"results": t}))
                        
                        
                        
                except Exception as e:
                    print("DataChannel send error:", e)

                # 6) 결과 프레임 교체
                new_frame = VideoFrame.from_ndarray(img_drawn, format="bgr24")
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
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(description)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}