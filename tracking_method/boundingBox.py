from itemStatus import ItemStatus
import cv2

def draw_bounding_box(imgs,r_o_c):
    color = ()
    for k,v in r_o_c.items():
        if k == ItemStatus.NORMAL.value:
            color = (0,255,0)
        elif k == ItemStatus.PENDING.value:
            color = (0,255,255)
        elif k == ItemStatus.MISPLACED.value:
            color = (0,0,255)
        else:
            color = (255, 255, 255)    
        for j in v:
            x1, y1, x2, y2 = map(int, j.xyxy)
            label = f"{j.tracker_id}"
            # 1. 내부 반투명 박스 (alpha blending 직접 적용)
            alpha = 0.25
            sub_img = imgs[y1:y2, x1:x2]                                # ROI
            overlay = sub_img.copy()   # 새로운 배열(메모리) 생성
            cv2.rectangle(overlay, (0,0), (x2-x1, y2-y1), color, -1)
            cv2.addWeighted(overlay, alpha, sub_img, 1 - alpha, 0, sub_img)

            # 2. 테두리 박스
            cv2.rectangle(imgs, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # 3. 텍스트 배경 + 텍스트
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            cv2.rectangle(imgs,
                        (x1, y1 - th - baseline),
                        (x1 + tw, y1),
                        color,
                        -1,
                        lineType=cv2.LINE_AA)

            cv2.putText(imgs,
                        label,
                        (x1, y1 - baseline),
                        font,
                        font_scale,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA)
