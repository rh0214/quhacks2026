import cv2
from ultralytics import YOLO
import time
from collections import Counter

cap = cv2.VideoCapture(0)
model = YOLO("best.pt")  # Load custom weights
model.fuse()

#timing for word collection
WINDOW_SECONDS = 2.0
window_start = time.time()
window_labels = []

final_words = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on this frame
    results = model(frame, conf=0.05, verbose=False)  # adjust confidence as needed

    # results[0] contains info for this frame
    annotated_frame = results[0].plot()  # draws boxes + labels
    cv2.imshow("Live ASL Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    r = results[0]
    print(time.time())
    if r.boxes is not None:
        for cls_id, conf in zip(r.boxes.cls, r.boxes.conf):
            label = r.names[int(cls_id)]
            print(f"Detected: {label} ({conf:.2f})")

cap.release()
cv2.destroyAllWindows()