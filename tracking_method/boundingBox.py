from tracking_method.itemStatus import ItemStatus
import cv2
from tracking_method.itemStatus import ItemStatus
import cv2

def draw_bounding_box(imgs, r_o_c, original_img, detections):
    current_ids = [tracker_id for tracker_id in detections.tracker_id]
  # 현재 시점 ID들만 추출

    for k, v in r_o_c.items():
        if k == ItemStatus.NORMAL.value:
            color = (0, 255, 0)
        elif k == ItemStatus.PENDING.value:
            color = (0, 255, 255)
        elif k == ItemStatus.MISPLACED.value:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)

        for j in v:
            if j.tracker_id not in current_ids:   # ✅ 현재 프레임에 없는 id는 스킵
                continue

            x1, y1, x2, y2 = map(int, j.xyxy)
            label = f"{j.tracker_id}"

            # 원본 해상도 보정
            orig_h, orig_w = original_img.shape[:2]
            resized_w, resized_h = (640, 640)
            x1 = int(x1 * orig_w / resized_w)
            x2 = int(x2 * orig_w / resized_w)
            y1 = int(y1 * orig_h / resized_h)
            y2 = int(y2 * orig_h / resized_h)

            # 1. 내부 반투명 박스
            alpha = 0.25
            sub_img = original_img[y1:y2, x1:x2]
            overlay = sub_img.copy()
            cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
            cv2.addWeighted(overlay, alpha, sub_img, 1 - alpha, 0, sub_img)

            # 2. 테두리 박스
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # 3. 라벨 텍스트
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            cv2.rectangle(original_img,
                          (x1, y1 - th - baseline),
                          (x1 + tw, y1),
                          color,
                          -1,
                          lineType=cv2.LINE_AA)

            cv2.putText(original_img,
                        label,
                        (x1, y1 - baseline),
                        font,
                        font_scale,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA)

    return original_img


