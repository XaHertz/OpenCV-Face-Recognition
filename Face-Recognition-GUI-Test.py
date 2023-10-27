from tkinter import *
from PIL import Image, ImageTk
import cv2

# create root window
root = Tk()
root.title("Face Recognition")
root.geometry('800x500')

# Create a Label to capture the Video frames
label = Label(root)
label.grid(row=0, column=0)

# Trained XML file for detecting faces
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)

   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)

   # Repeat after an interval to capture continiously
   label.after(20, show_frames)

show_frames()

# Execute Tkinter
root.mainloop()

