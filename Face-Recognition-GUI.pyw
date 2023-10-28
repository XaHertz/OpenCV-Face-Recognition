import os
import cv2
import numpy
import tkinter
import tkinter.font
import tkinter.messagebox
import tkinter.simpledialog
import PIL.Image

rawImagesPath = "dataset.images"
trainedPath = "dataset.trained"
if not os.path.exists(rawImagesPath):
    os.makedirs(rawImagesPath)
if not os.path.exists(trainedPath):
    os.makedirs(trainedPath)

faceDetectionCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Create root Window
root = tkinter.Tk()
root.title("Face Recognition")
root.geometry('400x400')
root.resizable(width=False, height=False)

def Add_Face_Button_command():
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    # For each person, enter one numeric face id and name
    face_id = tkinter.simpledialog.askstring(title="Enter User ID", prompt="Enter an User ID for Face Capture (Example: 00, 01, 02, ...)")
    id_name = tkinter.simpledialog.askstring(title="Enter User Name", prompt="Enter the Name of the User")
    with open(trainedPath + '/names.txt', 'a') as names:
        names.write(id_name)
        names.write('\n')

    tkinter.messagebox.showinfo(title='Initializing Face Capture', message='When you are ready to start press OK and Look at the Camera.')

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
            cv2.imshow('Face Capture', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break

    tkinter.messagebox.showinfo(title='Capture Complete', message='Face Data Captured Successfully.')

    # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()

def Train_Faces_Button_command():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = PIL.Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = numpy.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = faceDetectionCascade.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    faces, ids = getImagesAndLabels(rawImagesPath)
    recognizer.train(faces, numpy.array(ids))
    recognizer.write(trainedPath + '/trained.yml')

    # Print the numer of faces trained
    tkinter.messagebox.showinfo(title='Training Completed', message='Training Completed. {0} Faces Trained.'.format(len(numpy.unique(ids))))

def Recognize_Faces_Button_command():
    tkinter.messagebox.showinfo(title='Starting Recognizer', message='When you are ready press OK to start the Recognizer. When you are done use the ESC Button to Close the Recognizer Window.')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainedPath + '/trained.yml')

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # names related to ids
    with open(trainedPath + '/names.txt', 'r') as file:
        names = file.read().splitlines()

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
    
        cv2.imshow('Face Recognizer', img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()

Title_Label_CV=tkinter.Label(root)
Title_Label_CV["font"] = tkinter.font.Font(size=40)
Title_Label_CV["fg"] = "#333333"
Title_Label_CV["justify"] = "center"
Title_Label_CV["text"] = "OpenCV"
Title_Label_CV.place (x=10, y=40, width=380, height=60)

Title_Label_FR=tkinter.Label(root)
Title_Label_FR["font"] = tkinter.font.Font(size=24)
Title_Label_FR["fg"] = "#333333"
Title_Label_FR["justify"] = "center"
Title_Label_FR["text"] = "Face Recognition"
Title_Label_FR.place (x=10, y=100, width=380, height=60)

Add_Face_Button=tkinter.Button(root)
Add_Face_Button["bg"] = "#f0f0f0"
Add_Face_Button["font"] = tkinter.font.Font(size=10)
Add_Face_Button["fg"] = "#000000"
Add_Face_Button["justify"] = "center"
Add_Face_Button["text"] = "Add Face Data"
Add_Face_Button.place(x=100, y=210, width=200, height=30)
Add_Face_Button["command"] = Add_Face_Button_command

Train_Faces_Button=tkinter.Button(root)
Train_Faces_Button["bg"] = "#f0f0f0"
Train_Faces_Button["font"] = tkinter.font.Font(size=10)
Train_Faces_Button["fg"] = "#000000"
Train_Faces_Button["justify"] = "center"
Train_Faces_Button["text"] = "Train Faces"
Train_Faces_Button.place(x=100, y=260, width=200, height=30)
Train_Faces_Button["command"] = Train_Faces_Button_command

Recognize_Faces_Button=tkinter.Button(root)
Recognize_Faces_Button["bg"] = "#f0f0f0"
Recognize_Faces_Button["font"] = tkinter.font.Font(size=10)
Recognize_Faces_Button["fg"] = "#000000"
Recognize_Faces_Button["justify"] = "center"
Recognize_Faces_Button["text"] = "Recognize Faces"
Recognize_Faces_Button.place(x=100, y=310, width=200, height=30)
Recognize_Faces_Button["command"] = Recognize_Faces_Button_command

# Execute Tkinter
root.mainloop()
