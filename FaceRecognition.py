import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the Encoding File
print("Loading encodings...")
encodings = open("GeneratedEncodings.p", "rb")
encodingListKnownWithIDs = pickle.load(encodings)
encodings.close()
encodeListKnown, studentIDs = encodingListKnownWithIDs
print("Encodings loaded.")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)    

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS,faceCurrentFrame)

    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDistance)
        if matches[matchIndex]:
            print("Known Face Detected")
            print(studentIDs[matchIndex])


    cv2.imshow("Face Attendance", img)

    # Wait for a key press to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()