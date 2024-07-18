import cv2, numpy, os #cv-computer vision library,numpy-to perform array operations,os-to perform file directory operations
haar_file = 'haarcascade_fromtalface_default.xml' #algorithm xml file
face_cascade = cv2.CascadeClassifier(haar_file) #load of the haarcasde algotithm in a variable
datasets = 'datasets' #dataset folder name
print('Training...')
(images, labels, names, id)=([], [], {}, 0) #specific images & label in form of array and name in sets and id
for (subdirs, dirs, files) in os.walk(datasets): #this for loop for walk into the dataset folder
    for subdir in dirs: #this for loop for go into the subdirectory
        names[id] = subdir #assingn of the names to the sub directory
        subjectpath = os.path.join(datasets, subdir) #set of the path to the subdirectory 
        for filename in os.listdir(subjectpath): #this for loop for going into the subdirectory and get of the subject path
            path = subjectpath + '/' +filename 
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))#from these above lines we secregated all the folders
        id +=1
        
(images, labels)= [numpy.array(lis) for lis in [images, labels]] #images label everything in a single list
(width, height) = (130, 100)
model = cv2.face.LBPHFaceRecognizer_create() # classifier for accuracy and performance
#model = cv2.face.FisherFaceRecognizer_create() # another classifier

model.train(images, labels) #train of images and labels in the form ML concept
print(images,labels)
webcam = cv2.VideoCapture(0)
cnt = 0

#regonization part
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize) #this will predict the image
        cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 3) #to plot the rectangle box on the face
        if prediction[1]<800: #this is for the accuracy prediction
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.Font_HERSHEY_PLAIN,2,(0, 0, 255))
            print(names[prediction[0]]) #printing of the prdicted name
            cnt = 0
        else: #if continous resemble of unknown then marking it as different value
            cnt+=1
            cv2.putText(im,'unknown',(x-10, y-10), cv2.Font_HERSHEY_PLAIN,1,())
            if(cnt>100): #if count is more than 100 unknown person print and save the image 
                print("Unknown Person")
                cv2.imwrite("input.jpg",im)
                cnt=0
        cv2.imshow('FaceRecognition', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    webcam.release()



        
