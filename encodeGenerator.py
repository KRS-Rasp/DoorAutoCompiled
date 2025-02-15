import cv2
import face_recognition
import pickle
import os

### Importing the Faces
folderPath = "Faces"
pathList = os.listdir(folderPath)
imgList = []
studentIDs = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIDs.append(os.path.splitext(path)[0])

### Encodings Generator ###
def findEncodings(imagesList):
    encodeList  = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodingListKnownWithIDs = (encodeListKnown, studentIDs)
print("Encoding Complete.")

file = open("GeneratedEncodings.p", "wb")
pickle.dump(encodingListKnownWithIDs, file)
file.close()

print("Encodings Saved.")