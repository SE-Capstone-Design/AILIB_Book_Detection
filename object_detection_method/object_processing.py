from sklearn.cluster import DBSCAN 
import easyocr
import numpy as np
from sklearn.preprocessing import StandardScaler

ocr_Reader =  easyocr.Reader(['ko'], gpu = False) # 배포시에는 True

def get_object_detection_boxes(results):
    X = []

    for result in results:
        boxes = result.boxes  # box 객체

        if boxes is None or boxes.xyxy is None:
            continue

        for i in range(boxes.xyxy.shape[0]):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            X.append([int(x1), int(y1), int(x2), int(y2)])

    return np.array(X)


def extract_text_by_boxes_easyocr(image, boxes, reader=None):
    if reader is None:
        reader = ocr_Reader

    texts = []
    for box in boxes:
        id, x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            texts.append(None)
            continue

        result = reader.readtext(cropped, detail=0, paragraph=False)

        if not result:
            texts.append(None)
        else:
            texts.append(result[0].strip())
            
            
        # print(result)    
    return np.array(texts).reshape(-1, 1)


def cluster_boxes(boxes, eps=0.34990236345839165, min_samples=5):
    """
    boxes: List of bounding boxes [x1, y1, x2, y2]
    ocr_texts: OCR 결과 텍스트 리스트
    eps: DBSCAN의 거리 임계값 (같은 행으로 간주할 y축 거리)
    min_samples: 최소 샘플 수 (1로 하면 모든 점 포함 가능)
    
    return: 각 행에 대한 결과 
    """
    if boxes is None or len(boxes) == 0: # 예외처리
        return np.array([]).reshape(0, 1)    
    
    # 박스 하단 y좌표 계산
    X =boxes[:,2:4]
    X_scaler = StandardScaler().fit_transform(X)
    # print(y_centers)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaler)
    # clustering = HDBSCAN(min_cluster_size=2, store_centers = 'centroid' ).fit(X_scaler)
    labels = clustering.labels_  # 각 박스가 속한 행 번호
    # clustering.

    print("클러스터 라벨들:", sorted(set(labels)))
    return labels.reshape(-1,1)


def duplicate_text(combined):
    li = []  
    for label in np.unique(combined[:,0]):
        idx = np.where(combined[:, 0].astype(int) == label)[0]
        group = combined[idx]

        if len(group) < 3:
            continue

        unique_texts, counts = np.unique(group[:, 1], return_counts=True)

        # 모든 값이 동일한 그룹은 제외하고 싶다면
        if unique_texts.size == 1:
            continue

        # 최빈(동률 포함) 텍스트 집합
        most_common_texts = unique_texts[counts == counts.max()]

        # 최빈 텍스트들을 전부 제외 → “적게 나온 것들”만 남김
        keep_mask = ~np.isin(group[:, 1], most_common_texts)
        li.extend(group[keep_mask])

    return li
     

def ocr_with_row_clustering(image, results):
    boxes = get_object_detection_boxes(results)
    
    if boxes is None or len(boxes) == 0:
        return np.array([]) 
    
    texts = extract_text_by_boxes_easyocr(image, boxes)
    row_labels = cluster_boxes(boxes)

    combined = np.concatenate([row_labels.reshape(-1, 1), texts.reshape(-1, 1), boxes ], axis=1)
    # total = duplicate_text(combined)
    return combined  
