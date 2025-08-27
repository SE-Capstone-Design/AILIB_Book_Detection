
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
        """tracker_id 기준으로 우선 탐색해야 함.
            먼저 모든 row에 있는 아이템들 중 tracker_id 같은 게 있는지 확인
            있으면: 해당 아이템을 꺼내서 row가 바뀌었으면 새 row로 이동 및 OCR 신뢰도 비교후 높은 신뢰도로 ocr 교체 
            없으면: 그냥 새로 삽입"""
          # 1. tracker_id 탐색
        found_row, found_idx = None, None
        for row, items_in_row in self.items.items():
            for idx, it in enumerate(items_in_row):
                if it.tracker_id == item.tracker_id:
                    found_row, found_idx = row, idx
                    break
            if found_row is not None:
                break

        if found_row is not None:  
            # 2. 기존 item 존재
            old_item = self.items[found_row][found_idx]
            s = old_item.status  # 기존 status 유지

            # OCR 신뢰도 비교
            if old_item.ocr_conf is None and item.ocr_conf is None:
                # 둘 다 OCR 없음 → 좌표만 갱신
                old_item.xyxy = item.xyxy

            elif old_item.ocr_conf is None:
                # 기존 OCR 없음 → 새 item으로 교체
                self.items[found_row][found_idx] = item

            elif item.ocr_conf is None:
                # 새 item에 OCR 없음 → 좌표만 갱신
                old_item.xyxy = item.xyxy

            elif item.ocr_conf > old_item.ocr_conf:
                # 새 item이 OCR 더 정확 → 교체
                self.items[found_row][found_idx] = item
            else:
                # 기존 OCR 더 정확 → 좌표만 갱신
                old_item.xyxy = item.xyxy

            # status는 항상 유지
            self.items[found_row][found_idx].status = s

            # 3. row 변경되었으면 새 row로 이동
            if item.row != found_row:
                # 기존 row에서 제거
                moved_item = self.items[found_row].pop(found_idx)
                if not self.items[found_row]:
                    del self.items[found_row]  # 비면 삭제
                # 새 row에 삽입
                if item.row not in self.items:
                    self.items[item.row] = []
                self.items[item.row].append(moved_item)

        else:
            # 4. tracker_id 못 찾으면 새로 삽입
            if item.row not in self.items:
                self.items[item.row] = []
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
            parsed_indices = [(idx, it.parsed) 
                  for idx, it in enumerate(new_items) 
                  if it.parsed is not None]

            parsed_keys = [p for _, p in parsed_indices]
            lis_idx = self._lis_indices(parsed_keys)

            # lis_idx는 parsed_keys 인덱스 기준이니까,
            # 다시 new_items 인덱스로 변환
            lis_newitem_indices = {parsed_indices[i][0] for i in lis_idx}

            # LIS 안에 포함 → 정상, LIS 밖 → 잘못 배치
            for idx, it in enumerate(new_items):
                if it.parsed is None:
                    it.status = ItemStatus.PENDING
                elif idx in lis_newitem_indices:   # ✅ 이제 매핑 정확함
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
            
      
                    
                  
                
        
            
               
    
        
             
    
         
    
    