import cv2
from ultralytics import YOLO
import time
from collections import Counter
import requests
import json



# load models

cap = cv2.VideoCapture(0)

model1 = YOLO("phrase weights/best-2.pt")
model1.fuse()

model2 = YOLO("letter weights/best2.pt")
model2.fuse()



# llama integration
def correct_sentence_llama(raw_text):
    payload = {
        "model": "llama3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a grammar correction engine. "
                    "Rewrite the input into a natural, grammatically correct English sentence. "
                    "Do not add or remove meaning. "
                    "Do not explain. "
                    "Only output the corrected sentence."
                )
            },
            {
                "role": "user",
                "content": raw_text
            }
        ],
        "temperature": 0.1,
        "stream": False  # <--- THIS IS THE KEY FIX
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=15
    )

    # Check if the request actually worked
    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "").strip()
    else:
        return f"Error from Ollama: {response.text}"

# toggle + state

current_model = model1
model_name = "Phrases (Model 1)"

WINDOW_SECONDS = 1.5
window_start = time.time()
window_labels = []
final_words = []

quit_by_user = False   # <-- IMPORTANT FLAG


print("Starting detection...")
print("Press 'SPACE' to switch models | Press 'q' to quit.")



# main CV loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run inference
    if current_model == model1:
        results = model1(frame, conf=0.35, verbose=False)
    else:
        results = model2(frame, conf=0.3, verbose=False)

    annotated_frame = results[0].plot()

    # UI overlay
    cv2.putText(
        annotated_frame,
        f"Active: {model_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # detections
    r = results[0]
    if r.boxes is not None:
        for cls_id in r.boxes.cls:
            label = r.names[int(cls_id)]
            window_labels.append(label)

    # window capture
    current_time = time.time()
    if current_time - window_start >= WINDOW_SECONDS:
        if window_labels:
            most_common_word, _ = Counter(window_labels).most_common(1)[0]
            final_words.append(most_common_word)
            print(f"Captured ({model_name}): {most_common_word}")

        window_labels = []
        window_start = current_time

    # show frame
    cv2.imshow("Live ASL Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # quit
    if key == ord('q'):
        quit_by_user = True
        break

    # switch models
    elif key == 32:
        if current_model == model1:
            current_model = model2
            model_name = "Letters (Model 2)"
        else:
            current_model = model1
            model_name = "Phrases (Model 1)"
        print(f"\n>>> Switched to {model_name} <<<\n")



# cleanup

cap.release()
cv2.destroyAllWindows()



# run llama once (after quit)

if quit_by_user and final_words:
    raw_sentence = " ".join(final_words)

    print("\nRaw ASL Output:")
    print(raw_sentence)

    try:
        corrected_sentence = correct_sentence_llama(raw_sentence)
        print("\nCorrected English Sentence:")
        print(corrected_sentence)
    except Exception as e:
        print("\nllamA correction failed:")
        print(e)
else:
    print("\nNo sentence to correct.")
