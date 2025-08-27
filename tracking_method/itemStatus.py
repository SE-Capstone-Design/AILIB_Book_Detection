# tracking_id:int, xyxy: [], 'ocr':object,'ocr_conf':float , row: int, status: str, 
from enum import Enum
class ItemStatus(Enum):
    NEW = "new"    
    NORMAL = "normal"
    MISPLACED = "misplaced"
    PENDING = "pending"