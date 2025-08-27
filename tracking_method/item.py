from tracking_method.itemStatus import ItemStatus
from tracking_method.label_parsing import parse_ddc_key
import time

class Item:
    def __init__(self, tracker_id, xyxy, ocr, ocr_conf, row):
        self.tracker_id = tracker_id    
        self.xyxy = xyxy
        self.ocr = ocr # 없을경우는 기본적 입력할때 None임 
        self.ocr_conf = ocr_conf  # None 또는 float
        self.row = int(row)
        self.status = ItemStatus.NEW if ocr_conf > 0.7 else ItemStatus.PENDING  #ocr 신뢰도가 0.7미만인 경우 보류로 설정 
        self.parsed =parse_ddc_key(ocr) if ocr is not None else None # 없을경우는 None으로 
        self.updated_at = time.time()

    
    def x1(self):
        return float(self.xyxy[0])
    
    def id(self):
        return self.tracker_id
    
    def to_dict(self):
        return {
            "tracker_id": self.tracker_id,
            "xyxy": self.xyxy,
            "ocr": self.ocr,
            "ocr_conf": self.ocr_conf,
            "row": self.row,
            "status": self.status.name if self.status else None
        }