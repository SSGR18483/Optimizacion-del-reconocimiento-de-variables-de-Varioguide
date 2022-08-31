from tkinter import*
import cv2
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import pytesseract as tess

import tkinter as tk
from tkinter import Tk, Label, Button
from PIL import ImageTk,Image 
#import serial, time
#arduino=serial.Serial('COM1',9600)
#time.sleep(2) 



#cam = cv2.VideoCapture(1)
coso=0
tess.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img_counter = 0
def myClick():
    # ret, frame = cam.read()
    # img_counter =  0
    # hello=""+a.get()
    # myLabel=Label(root,text=hello)
    # #myLabel=Label(root,text=e.get())
    # myLabel.place(x=1000, y=100)
    
   
    # img_name = "opencv_frame_11.png".format(img_counter)
    # cv2.imwrite(img_name, frame)
    # print("{} written!".format(img_name))
    # img_counter += 1 
    # img3 = ImageTk.PhotoImage(Image.open("opencv_frame_11.png"))
    
   
    img = Image.open("caj5.png")
    img = img.resize((500,300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)  
      
   
    label2=Label(image=img)
    label2.image = img
    label2.place(x=700, y=400)
    
    
    # img=Image.open("humana1.png")
    # print(pytesseract.image_to_string(img))
   
    # img2 = cv2.imread("humana1.png")
    # h, w, c = img2.shape
    # boxes = pytesseract.image_to_boxes(img2) 
    # text = pytesseract.image_to_string(img2) 
    # for b in boxes.splitlines():
    #     b = b.split(" ")
    #     img4 = cv2.rectangle(img2, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    # cv2.imshow("img", img4)
    # cv2.waitKey(0)
    # print(text)
  
    
# def video_stream():
#     ret, frame = cam.read()
#     frame=cv2.resize(frame,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(1, video_stream) 
    
        
      
        
    
while True:
    # ret, frame = cam.read()
    # if not ret:
    #     print("failed to grab frame")
        
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == 27:
        
    #     # ESC pressed
    #     print("Escape hit, closing...")
        
    # elif cv2.waitKey(1) & 0xFF == 32:  
    #     # SPACE pressed
    #     img_name = "opencv_frame_11.png".format(img_counter)
    #     cv2.imwrite(img_name, frame)
    #     print("{} written!".format(img_name))
    #     img_counter += 1 
    root = Tk()  
    mys = PhotoImage(file = r"C:\Users\joe\caj1.png")
    mys2 = PhotoImage(file = r"C:\Users\joe\caj2.png")
   
    app = Frame(root, bg="white")
    app.grid()
    # Create a label in the frame
    lmain = Label(app)
    lmain.grid(row = 6, column = 0, sticky = W, pady = 500)
    
    
    
   

    canvas = Canvas(root, width = 650, height = 500)  
    canvas2 = Canvas(root, width = 300, height = 300)  
   
    img = Image.open("caj2.png")
    img = img.resize((2000,1000), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)  
    img2 = Image.open("caj1.png")
    img2 = img2.resize((500,500), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(img2) 
    img3 = Image.open("caj4.png")
    img3 = img3.resize((350,150), Image.ANTIALIAS)
    img3 = ImageTk.PhotoImage(img3)   
   
    label2=Label(image=img)
    label2.image = img
    label2.place(x=1, y=1)
    
    label3=Label(image=img3)
    label3.image = img3
    label3.place(x=775, y=400)
    
    # video_stream()
    
    

    
    #e.pack()  
    # a =tk.Entry(root,bg="white",fg="red",width=50)
    # a.insert(0,"porcentaje de error es :   ") 
    # a.place(x=1400, y=270)
    
    # b =tk.Entry(root,bg="white",fg="red",width=50)
    # b.insert(0,"porcentaje de error es :   ") 
    # b.place(x=1400, y=530)
    
    # c =tk.Entry(root,bg="white",fg="red",width=50)
    # c.insert(0,"porcentaje de error es :   ") 
    # c.place(x=1400, y=720)

  
    
    myButton=Button(root,image=img2,compound = LEFT,command=myClick)
    myButton.place(x=200, y=200) 
    
    myButton2=Button(root,image=img2,compound = LEFT,command=myClick)
    myButton2.place(x=1200, y=200) 
  
   
    #myButton.pack()

    root.mainloop()   
        
    cam.release()             

    cv2.destroyAllWindows()                
    break