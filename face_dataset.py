import cv2
import os
dataset = "dataset"
name = "captured_photos"
path = os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)

(width,height) = (130,100)  
haar_file = 'haarcascade_frontalface_default.xml'    
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count<31:
    print(count)
    _,im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        faceonly = gray[y:y+h,x:x+h]
        resizeImg = cv2.resize(faceonly,(width,height))
        cv2.imwrite("%s/%s.jpg" %(path,count),resizeImg)
        count+=1
    cv2.imshow('FaceDetection', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
print("Image Captured successfully")
webcam.release()
cv2.destroyAllWindows()
