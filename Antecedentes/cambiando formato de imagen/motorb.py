import cv2
from PIL import Image
import re
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import serial, time
import struct
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"


contador = bytes(111)
indice = 0
#buf[indice] = 0
# pasos = 100
# buffer = []
# recibiendo = False

# port_arduino = 'COM5'
# port_tiva = 'COM4'

# ser  = serial.Serial(port=port_tiva,baudrate=115200)
# ser.flushInput()



numero=[]
image=cv2.imread("new_imageb.png")
newImg = cv2.imread("new_imageb.png")
# newImg =cv2.imread("new_imageb.png")






gray=cv2.cvtColor(newImg,cv2.COLOR_BGR2GRAY)

gray=cv2.blur(gray,(2,2))
the=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
canny=cv2.Canny(the,0,200)


canny2=cv2.dilate(canny,None,iterations=1)

cnts,cnts1=cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image,cnts,171,(0,255,0),2)
text=pytesseract.image_to_string(newImg,config="--psm 11")
texta=pytesseract.image_to_string(gray,config="--psm 11")
textb=pytesseract.image_to_string(canny,config="--psm 11")
textc=pytesseract.image_to_string(canny2,config="--psm 11")
print(text.find("-"))
a=text.replace(".","")
aa=a.replace(",","")
textraf=aa.replace("°","")

b=len(textraf)
print(b)
v1=0
v2=0
v3=0
v4=0
if(b<=5 and v1==0):
    v1=1
    
if(b>=6 and v1==0):
    
    #print("v1",v1)
    textraf=textraf[0:7]
   
if(b>=5 and v1!=1):
    
    #print("v2",v2)
    textraf=textraf[0:6]   
    
if(b>=4 and v2!=1 and v1!=1):
    
    #print("v3",v3)
    textraf=textraf[0:5] 
   


if(text[0]=="-" and text[2]=="."):
    print("textraF:",textraf)
    a1=textraf[0:1]
    a2=textraf[1:3]
    cadenasa1=[a1,"0",a2]
    textraf="".join(cadenasa1)
    v4=1 
if(text[0]=="-" and text[2]==","):
    a1=textraf[0:1]
    a2=textraf[1:3]
    cadenasa1=[a1,"0",a2]
    textraf="".join(cadenasa1)
    v4=1  
if(text[0]=="-" and v4==0):
    a1=textraf[0:1]
    a2=textraf[1:3]
    cadenasa1=[a1,"0",a2]
    textraf="".join(cadenasa1)
    v4=0
if(text[0]!="+" and text[0]!="-"):
    
    cadenasa2=["+",textraf]
    textraf="".join(cadenasa2)
   
    
    if(textraf[0]=="+"  and text[1]=="."):
        a3=textraf[0]
        a4=textraf[1:3]
        cadenasa3=[a3,"0",a4]
        v4=1 
        textraf="".join(cadenasa3)
    if(textraf[0]=="+"  and text[1]==","):
        a3=textraf[0]
        a4=textraf[1:3]
        cadenasa3=[a3,"0",a4]
        textraf="".join(cadenasa3)
        v4=1 
    if(textraf[0]=="+"  and v4==0):
        a3=textraf[0]
        a4=textraf[1:3]
        cadenasa3=[a3,"0",a4]
        textraf="".join(cadenasa3)
        v4=1 
   
    
    
       
cadenasa4=["h","S",textraf,"Z"]
textrafb="".join(cadenasa4)        
print(textrafb)


# string to bytes using bytes()
textrafb1 = bytes(textrafb, encoding='utf-8')
print(type(textrafb1))







# try:
#     while (ser.isOpen()):
#         #if not ser.isOpen():
#         #    ser.open()
        
#         ser.write(textrafb1)
        
   
        
#         #if (ser.read() >= str('a').encode('ascii')):
                
        
#         datos = ser.read()
#         print(datos)
#         if (datos == str('O').encode('ascii')):
#             buffer.append(datos)
#             for i in range(1,2):
#                 val = ser.read()
#                 buffer.append(val)
#             ok = b''.join(buffer)
#             print(ok)
#             buffer = []
#         if (ok == b'OK'):
#             print("Handshake exitoso")
                 
#         recv = ser.read(7)
#         print(recv)
        
        
        
#         time.sleep(0.02)
        
# except KeyboardInterrupt:
#     ser.close()
# finally:
#     ser.close()





# cv2.imshow("image",image)
# cv2.imshow("re",newImg)
# cv2.imshow("gray",gray)
# cv2.imshow("canny",canny)
# cv2.imshow("the",the)

# cv2.moveWindow("image",45,10)
# cv2.waitKey(0)


d = pytesseract.image_to_data(newImg, output_type=Output.DICT)


# date_pattern = "-0.1°"

# n_boxes = len(d["text"])
# for i in range(n_boxes):
#     if float(d["conf"][i]) < 1000:
#      	if re.match(date_pattern, d["text"][i]):
#  	        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
#  	        img = cv2.rectangle(newImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow("img", img)
# cv2.waitKey(0)