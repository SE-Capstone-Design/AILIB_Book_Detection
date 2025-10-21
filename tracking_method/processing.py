from sklearn.cluster import DBSCAN 
import easyocr
import numpy as np
import cv2

# ocr_Reader =  easyocr.Reader(['ko','en'], gpu = False,model_storage_directory= None,detector =False,quantize =False,cudnn_benchmark =True) # 배포시에는 True
ocr_Reader = easyocr.Reader(['ko','en'], gpu=True)

def get_object_detection_boxes(detections):
    X = []
    # tracker_id, bbox 좌표 추출

    for tracker_id, xyxy, confidence in zip(detections.tracker_id, detections.xyxy,detections.confidence):
        if tracker_id is None:
            continue
        X.append({"tracker_id" : int(tracker_id), "xyxy": xyxy.tolist(),"confidence": confidence})
    return X
    
    

def extract_text_by_boxes_easyocr(original_img, boxes_id, reader=None):
    """
    original_img: 원본 이미지 webrtc_frame
    boxes_id: YOLO 추적 후 나온 박스 리스트 (리사이즈된 이미지 기준 좌표)
    resized_shape: YOLO 모델 입력 크기 (기본 640x640)
    """
    if reader is None:
        reader = ocr_Reader
    
    
    # gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    original_img = enhance_img(original_img)    
    
      
    for i, box in enumerate(boxes_id):
        x1, y1, x2, y2 = map(int, box["xyxy"])

        cropped = original_img[y1:y2+1, x1:x2+1]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)


        if cropped.size == 0:
            boxes_id[i]['ocr'] = None
            boxes_id[i]['ocr_conf'] = None
            continue

        # detail=1 → [[bbox, text, conf], ...]
        result = reader.readtext(cropped, detail=1, paragraph=False)
        
        # result = ocr.predict(cropped)
        for r in result:
            texts = r['rec_texts']
            scores = r['rec_scores']
            polys = r['rec_polys']   # 위치 좌표
            
            for text, conf, poly in zip(texts, scores, polys):
                print(f"Text: {text}, Conf: {conf:.4f}, Poly: {poly}")    
                

                if not text:
                    boxes_id[i]['ocr'] = None
                    boxes_id[i]['ocr_conf'] = 0.0
                else:
                    _, text, conf = result[0]
                    boxes_id[i]['ocr'] = text.strip()
                    boxes_id[i]['ocr_conf'] = float(conf)

                    print("test:", text.strip(), "conf:", conf)
            
            # if not result:
            #     boxes_id[i]['ocr'] = None
            #     boxes_id[i]['ocr_conf'] = 0.0
            # else:
            #     _, text, conf = result[0]
            #     boxes_id[i]['ocr'] = text.strip()
            #     boxes_id[i]['ocr_conf'] = float(conf)

            #     print("test:", text.strip(), "conf:", conf)

    return boxes_id



def cluster_boxes(boxes, eps=800, min_samples=1):
    """
    boxes: List of bounding boxes [x1, y1, x2, y2]
    ocr_texts: OCR 결과 텍스트 리스트
    eps: DBSCAN의 거리 임계값 (같은 행으로 간주할 y축 거리)
    min_samples: 최소 샘플 수 (1로 하면 모든 점 포함 가능)
    
    return: 각 행에 대한 결과 
    """
     # 박스 하단 y좌표 계산
    y_centers = np.array([
    [(box['xyxy'][0] + box['xyxy'][2]) / 2,   # x_center
     (box['xyxy'][1] + box['xyxy'][3]) / 2]   # y_center
    for box in boxes
])
#     y_centers = np.array([
#     (box['xyxy'][1] + box['xyxy'][3]) / 2   # (y1 + y2) / 2
#     for box in boxes
# ]).reshape(-1, 1)
    
  
    # print(y_centers)
    # DBSCAN으로 y축 클러스터링 → 같은 y좌표 = 같은 행
    clustering = DBSCAN(eps=eps, min_samples=min_samples,).fit(y_centers)
    labels = clustering.labels_  # 각 박스가 속한 행 번호

    for i in range(len(boxes)):
        boxes[i]['row'] = labels[i]
    
    return boxes



def row_ocr_clustering(tracked,original_img):
    boxes_id = get_object_detection_boxes(tracked) 
    if boxes_id is None or len(boxes_id) == 0:
        return [] 
    ocr_boxes = extract_text_by_boxes_easyocr(original_img, boxes_id)
    row_ocr_boxes = cluster_boxes(ocr_boxes)
    return row_ocr_boxes


def enhance_img(img):
    print("shape:",img.shape)
    print("dim",img.ndim)
    print("frame shape:", img.shape, "dtype:", img.dtype)
    print("frame channels:", 1 if len(img.shape)==2 else img.shape[2])
    img = img.copy()
    #1. 2.해상도 키우기 (3~4배 정도)
    img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    # 3 그레이스케일 변환
      # ⚠️ BGR -> GRAY 변환
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img 
    
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 4 대비 강화
    # gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)  # alpha=1.8~2.5로 조절 가능
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.addWeighted(gray, 2.5, blur, -0.8, 0) 
    
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 15, 5)
    
    #  글자 두껍게 (Morphology)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    proc = cv2.dilate(th, kernel, iterations=1)
    return proc
