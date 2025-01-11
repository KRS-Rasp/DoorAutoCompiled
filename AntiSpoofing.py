from ultralytics import YOLO
import cv2
import cvzone
import math
import time

confidence = 0.9

cap = cv2.VideoCapture(0) # for webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("test.mp4") # for video file

model = YOLO("models/best.pt")
model.overrides['verbose'] = False

classNames = ["fake", "real"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Confidence
            conf = math.ceil((box.conf[0] * 100))/100
            # class name
            if conf > confidence:
                cls = int(box.cls[0])
                if classNames[cls] == "real":
                    print("Real")
                else:
                    print("Spoof")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    # Exit if 'q' key is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()