import cv2
import numpy as np
from PIL import Image
import os

rawImagesPath = "dataset.images"
trainedPath = "dataset.trained"
if not os.path.exists(rawImagesPath):
   os.makedirs(rawImagesPath)
if not os.path.exists(trainedPath):
   os.makedirs(trainedPath)

faceDetectionCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml');

print('\n [START] Welcome to OpenCV Face Recognition Program!\n')
userChoice = 0

def capture():
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    # For each person, enter one numeric face id
    face_id = input('\nEnter a User ID = ')

    print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

    # Initialize individual sampling face count
    count = 0

    while(True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetectionCascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite(rawImagesPath + "/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break

    print(" [INFO] Face data captured.\n")

    # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()

def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = faceDetectionCascade.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    print ("\n [INFO] Training faces. It will take a few seconds. Please Wait ...")

    faces, ids = getImagesAndLabels(rawImagesPath)
    recognizer.train(faces, np.array(ids))
    recognizer.write(trainedPath + '/trained.yml')

    # Print the numer of faces trained
    print(" [INFO] {0} faces trained.\n".format(len(np.unique(ids))))

def recognize():
    print("\n [INFO] Recognizing Faces. Use ESC to Close the Recognizer Window.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainedPath + '/trained.yml')

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # names related to ids
    names = ['None', 'Akshat', 'Hemant', 'Vidhan', 'Z', 'W']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetectionCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
        cv2.imshow('camera', img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    print(" [INFO] Recognizer Window closed.\n")

    # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()

while userChoice != '4':
    print('Enter A Choice:\n1. Add Face Data\n2. Train Faces\n3. Recognize Faces\n4. Exit\n')
    userChoice = input('Choice = ')
    if userChoice == '1':
        capture()
    elif userChoice == '2':
        train()
    elif userChoice == '3':
        recognize()
    elif userChoice == '4':
        print("\n [END] Exiting Program.\n")
    else:
        print("\n [ERROR] Choice Invalid! Please Retry!\n")
