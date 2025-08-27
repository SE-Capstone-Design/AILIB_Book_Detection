
from tracking_method.itemStatus import ItemStatus
from tracking_method.item import Item

def _lis_indices(keys):
    """O(n log n) LIS: 증가 부분수열의 '원본 인덱스 집합'을 반환."""
    tails = []               # 길이 k 수열의 '최소 꼬리' 원소 인덱스
    prev = [-1] * len(keys)  # 재구성 포인터

    def less(i, j):
        return keys[i].tup() < keys[j].tup()

    for i in range(len(keys)):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if less(tails[mid], i):
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i
        if lo > 0:
            prev[i] = tails[lo - 1]

    res = []
    cur = tails[-1] if tails else -1
    while cur != -1:
        res.append(cur)
        cur = prev[cur]
    return set(reversed(res))

def _to_float_or_none(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None
    
   
   
class ManageItem:
    def __init__(self): 
        self.items ={} # Key: Item.row value: [Item class]  
    
    def insert_update(self, item:Item):
        """Item을 row 기준으로 items에 추가한다"""
        if item.row not in self.items:
            self.items[item.row] = []       
        
        ## tracker_id 중복 여부 확인 
        for i,it in enumerate(self.items[item.row]):
            # 만약 tracker_id가 같은 경우 ocr 신뢰도 높은 값으로 update 
            # 아닐 경우 삽입 
            if item.tracker_id == it.tracker_id:
                if it.ocr_conf is None and item.ocr_conf is None:
                    return 
                elif it.ocr_conf is None:
                    s = self.items[item.row][i].status
                    self.items[item.row][i] = item
                    self.items[item.row][i].status = s                      
                elif item.ocr_conf is None: 
                    return                    
                elif item.ocr_conf > it.ocr_conf:
                    s = self.items[item.row][i].status
                    self.items[item.row][i] = item
                    self.items[item.row][i].status = s                      
                return    
                    
        self.items[item.row].append(item)     
    
    def input_predict(self, results): ## 예측 결과를 items 변수에 삽입
        for i in results:
            self.insert_update(Item(tracker_id=i.get('tracker_id'),
                xyxy=i.get('xyxy'),
                ocr=i.get('ocr'),
                ocr_conf=i.get('ocr_conf'),
                row=i.get('row')
            ))
    
        
    def _lis_indices(self, keys):
        """O(n log n) LIS: 증가 부분수열의 인덱스 집합을 반환."""
        tails = []
        prev = [-1] * len(keys)

        def less(i: int, j: int) -> bool:
            return keys[i].tup() < keys[j].tup()

        for i in range(len(keys)):
            lo, hi = 0, len(tails) # 초기화 low : 0 index high : 입력된 len(list) 
            while lo < hi:
                mid = (lo + hi) // 2
                if less(tails[mid], i):
                    lo = mid + 1
                else:
                    hi = mid
            if lo == len(tails):
                tails.append(i)
            else:
                tails[lo] = i
            if lo > 0:
                prev[i] = tails[lo - 1]

        res = []
        cur = tails[-1] if tails else -1
        while cur != -1:
            res.append(cur)
            cur = prev[cur]
        return set(reversed(res))         
                
    def detect_bookLabel(self):        
        for k, v in self.items.items():
            if not v: # 데이터가 없을 시 판정 X (혹시모를 예외처리)
                continue
            if len(v) < 3: # 2개이하일때는 판정을 안함(보류 판정)
                for j in self.items[k]:
                    j.status = ItemStatus.PENDING  # 보류 판정
                continue
            
            new_items = [it for it in v if it.status == ItemStatus.NEW or it.status == ItemStatus.MISPLACED]
            if not new_items: #없을경우 해당 행 종료 
                continue

            # DDC 키 기준 정렬 (x좌표 순서)
            new_items = sorted(new_items, key=lambda x: float(x.xyxy[0]))

            # parsed (DdcKey) 추출
            parsed_keys = [it.parsed for it in new_items if it.parsed is not None]

            # LIS 인덱스 구하기
            lis_idx = self._lis_indices(parsed_keys)

            # LIS 안에 포함 → 정상, LIS 밖 → 잘못 배치
            for idx, it in enumerate(new_items):
                if it.parsed is None:
                    it.status = ItemStatus.PENDING
                elif idx in lis_idx:
                    it.status = ItemStatus.NORMAL
                else:
                    it.status = ItemStatus.MISPLACED        
    
    def get_misplaced(self):
        wrong = []  
        for i in self.items.values():
            for v in i:
                if v.status == ItemStatus.MISPLACED:
                    wrong.append(v)       
        return wrong
    
    def get_pending(self) :
        pending = [] 
         
        for i in self.items.values():
            for v in i:
                if v.status == ItemStatus.PENDING:
                    pending.append(v)       
        return pending   
    
    def get_normal(self):
        normal = []
        for i in self.items.values():
            for v in i:
                if v.status == ItemStatus.NORMAL:
                    normal.append(v)       
        return normal         
            
     
    def start(self,results):
        self.input_predict(results)
        self.detect_bookLabel()
        n = self.get_normal()
        m = self.get_misplaced()
        p = self.get_pending()

        return {"normal": n, "misplaced":m,"pending":p }
            
      
                    
                  
                
        
            
               
    
        
             
    
         
    
    