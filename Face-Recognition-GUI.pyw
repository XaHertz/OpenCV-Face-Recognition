import os
import cv2
import numpy
import tkinter
import tkinter.font
import tkinter.messagebox
import tkinter.simpledialog
import PIL.Image
import PIL.ImageTk

rawImagesPath = "dataset.images"
trainedPath = "dataset.trained"
if not os.path.exists(rawImagesPath):
    os.makedirs(rawImagesPath)
if not os.path.exists(trainedPath):
    os.makedirs(trainedPath)

faceDetectionCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

root = tkinter.Tk()
root.title("Face Recognition")
root.geometry('952x522')
root.resizable(width=False, height=False)

def INIT_Camera_Window():
    cam = cv2.VideoCapture(0)
    def show_frames():
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetectionCascade.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        colored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgx = PIL.Image.fromarray(colored)
        imgtk = PIL.ImageTk.PhotoImage(image = imgx)
        Camera_Window.imgtk = imgtk
        Camera_Window.configure(image=imgtk)
        Camera_Window.after(20, show_frames)
    show_frames()

def Add_User():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    user_id = tkinter.simpledialog.askstring(title="Enter User ID", prompt="Enter an User ID for Face Capture (Example: 00, 01, 02, ...)")
    user_name = tkinter.simpledialog.askstring(title="Enter User Name", prompt="Enter the Name of the User")
    with open(trainedPath + '/names.txt', 'a') as names:
        names.write(user_name)
        names.write('\n')
    tkinter.messagebox.showinfo(title='Initializing Face Capture', message='When you are ready to start press OK and Look at the Camera.')
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetectionCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            cv2.imwrite(rawImagesPath + "/User." + str(user_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            #cv2.imshow('Face Capture', img)
        #k = cv2.waitKey(100) & 0xff
        #if k == 27:
        #    break
        #elif count >= 30:
        #    break
        if count >= 30:
            break
    tkinter.messagebox.showinfo(title='Capture Complete', message='User Data Captured Successfully. Please Retrain Faces!')
    cam.release()
    cv2.destroyAllWindows()
    if os.path.exists(trainedPath + '/trained.yml'):
        Recognize_Faces()
    else:
        INIT_Camera_Window()

def Remove_User():
    user_id = tkinter.simpledialog.askstring(title="Enter User ID", prompt="Enter an User ID to be Removed (Example: 00, 01, 02, ...)")
    with open(trainedPath + '/names.txt', 'r') as file:
        names = file.read().splitlines()
    names.pop(int(user_id))
    with open(trainedPath + '/names.txt', 'w') as file:
        for name in names:
            file.write(name)
            file.write('\n')
    for count in range(30):
        os.remove(rawImagesPath + "/User." + str(user_id) + '.' + str(count+1) + ".jpg")
    tkinter.messagebox.showinfo(title='Removal Complete', message='User Data Removed Successfully. Please Retrain Faces!')

def Train_Faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = PIL.Image.open(imagePath).convert('L')
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
    tkinter.messagebox.showinfo(title='Training Completed', message='Training Completed. {0} Faces Trained.'.format(len(numpy.unique(ids))))

def Recognize_Faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainedPath + '/trained.yml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    with open(trainedPath + '/names.txt', 'r') as file:
        names = file.read().splitlines()
    cam = cv2.VideoCapture(0)
    def show_frames():
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetectionCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (64, 48))
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        colored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgx = PIL.Image.fromarray(colored)
        imgtk = PIL.ImageTk.PhotoImage(image = imgx)
        Camera_Window.imgtk = imgtk
        Camera_Window.configure(image=imgtk)
        Camera_Window.after(20, show_frames)
    show_frames()

Title_Label_CV = tkinter.Label(root)
Title_Label_CV["font"] = tkinter.font.Font(size=40)
Title_Label_CV["fg"] = "#333333"
Title_Label_CV["justify"] = "center"
Title_Label_CV["text"] = "OpenCV"
Title_Label_CV.place(x=20, y=40, width=250, height=60)

Title_Label_FR = tkinter.Label(root)
Title_Label_FR["font"] = tkinter.font.Font(size=24)
Title_Label_FR["fg"] = "#333333"
Title_Label_FR["justify"] = "center"
Title_Label_FR["text"] = "Face\nRecognition"
Title_Label_FR.place(x=20, y=100, width=250, height=80)

Add_User_Button = tkinter.Button(root)
Add_User_Button["bg"] = "#f0f0f0"
Add_User_Button["font"] = tkinter.font.Font(size=10)
Add_User_Button["fg"] = "#000000"
Add_User_Button["justify"] = "center"
Add_User_Button["text"] = "Add User"
Add_User_Button.place(x=50, y=220, width=190, height=30)
Add_User_Button["command"] = Add_User

Remove_User_Button = tkinter.Button(root)
Remove_User_Button["bg"] = "#f0f0f0"
Remove_User_Button["font"] = tkinter.font.Font(size=10)
Remove_User_Button["fg"] = "#000000"
Remove_User_Button["justify"] = "center"
Remove_User_Button["text"] = "Remove User"
Remove_User_Button.place(x=50, y=265, width=190, height=30)
Remove_User_Button["command"] = Remove_User

Train_Faces_Button = tkinter.Button(root)
Train_Faces_Button["bg"] = "#f0f0f0"
Train_Faces_Button["font"] = tkinter.font.Font(size=10)
Train_Faces_Button["fg"] = "#000000"
Train_Faces_Button["justify"] = "center"
Train_Faces_Button["text"] = "Train Faces"
Train_Faces_Button.place(x=50, y=310, width=190, height=30)
Train_Faces_Button["command"] = Train_Faces

Recognize_Faces_Button = tkinter.Button(root)
Recognize_Faces_Button["bg"] = "#f0f0f0"
Recognize_Faces_Button["font"] = tkinter.font.Font(size=10)
Recognize_Faces_Button["fg"] = "#000000"
Recognize_Faces_Button["justify"] = "center"
Recognize_Faces_Button["text"] = "Recognize Faces"
Recognize_Faces_Button.place(x=50, y=355, width=190, height=30)
Recognize_Faces_Button["command"] = Recognize_Faces

Camera_Window = tkinter.Label(root)
Camera_Window["bg"] = "#000000"
Camera_Window.place(x=290, y=20, width=640, height=480)
INIT_Camera_Window()

root.mainloop()
