import cv2
from ultralytics import YOLO
import time
from collections import Counter

# Load models
cap = cv2.VideoCapture(0)
model1 = YOLO("phrase weights/best-2.pt")
model1.fuse()
model2 = YOLO("letter weights/best.pt")
model2.fuse()

# Toggle setup
current_model = model1
model_name = "Phrases (Model 1)"

# Timing and storage configuration
WINDOW_SECONDS = 1.5
window_start = time.time()
window_labels = []
final_words = []

print("Starting detection...")
print("Press 'SPACE' to switch models | Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference with the ACTIVE model
    if current_model == model1:
        results = model1(frame, conf=0.35, verbose=False)
    else:
        results = model2(frame, conf=0.3, verbose=False)

    annotated_frame = results[0].plot()
    
    # UI Overlay to show which model is active
    cv2.putText(annotated_frame, f"Active: {model_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Collect classifications
    r = results[0]
    if r.boxes is not None:
        for cls_id in r.boxes.cls:
            label = r.names[int(cls_id)]
            window_labels.append(label)

    # Window logic
    current_time = time.time()
    if current_time - window_start >= WINDOW_SECONDS:
        if window_labels:
            most_common_word, count = Counter(window_labels).most_common(1)[0]
            final_words.append(most_common_word)
            print(f"Captured ({model_name}): {most_common_word}")
        
        window_labels = []
        window_start = current_time

    # Display the frame
    cv2.imshow("Live ASL Detection", annotated_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'q' to quit
    if key == ord('q'):
        break
    
    # Press 'SPACE' to switch models
    elif key == 32: 
        if current_model == model1:
            current_model = model2
            model_name = "Letters (Model 2)"
        else:
            current_model = model1
            model_name = "Phrases (Model 1)"
        print(f"\n>>> Switched to {model_name} <<<\n")

cap.release()
cv2.destroyAllWindows()

print("\nFinal Resulting Sentence:")
print(" ".join(final_words))