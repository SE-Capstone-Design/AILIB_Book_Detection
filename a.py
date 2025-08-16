import cv2, ctypes
from tracking_method.processing import get_object_detection_boxes , cluster_boxes
import numpy as np

from ultralytics import YOLO
# Load the YOLO11 model
model = YOLO("weights.pt")

# Open the video file
video_path = r"E:\다운로드\test_3.mp4"

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        boxes = get_object_detection_boxes(results)
    
        row_labels = cluster_boxes(boxes)
        
        # Display the annotated frame
        win = "YOLO11 Tracking"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # 수동 리사이즈 가능한 창
        cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

        # 화면 해상도 구하기 (Windows)
        user32 = ctypes.windll.user32
        screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        # ... 루프 안에서 annotated_frame 만든 뒤에:
        h, w = annotated_frame.shape[:2]

        # 여백(타이틀바 등) 고려한 최대 크기
        max_w = screen_w - 120
        max_h = screen_h - 160

        # 화면에 맞추는 스케일
        r = min(max_w / w, max_h / h, 1.0)
        if r < 1.0:
            disp = cv2.resize(annotated_frame, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)
        else:
            disp = annotated_frame

        cv2.imshow(win, disp)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()