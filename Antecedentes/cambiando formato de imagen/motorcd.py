import cv2
from PIL import Image
import re
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

numero=[]

imagec1=cv2.imread("selfie.png")
newImgc1 = cv2.imread("new_imagec1.png")


grayc1=cv2.cvtColor(newImgc1,cv2.COLOR_BGR2GRAY)

grayc1=cv2.blur(grayc1,(2,2))
thec1=cv2.threshold(grayc1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
cannyc1=cv2.Canny(thec1,0,200)


canny2c1=cv2.dilate(cannyc1,None,iterations=1)

cnts,cnts1=cv2.findContours(cannyc1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image,cnts,171,(0,255,0),2)
textc1=pytesseract.image_to_string(newImgc1,config="--psm 11")
textac1=pytesseract.image_to_string(grayc1,config="--psm 11")
textbc1=pytesseract.image_to_string(cannyc1,config="--psm 11")
textcc1=pytesseract.image_to_string(cannyc1,config="--psm 11")
#print(text.find("02°"))

bc1=len(textc1)
print(bc1)
v1c1=0
v2c1=0
v3c1=0
if(bc1<=5 and v1c1==0):
    v1c1=1
    
if(bc1>=6 and v1c1==0):
    
    print("v1c1",v1c1)
    textc1=textc1[0:7]
    print(textc1)
if(bc1>=5 and v1c1!=1):
    
    print("v2c1",v2c1)
    textc1=textc1[0:6]   
    print(textc1)
if(bc1>=4 and v2c1!=1 and v1c1!=1):
    
    print("v3c1",v3c1)
    textc1=textc1[0:5] 
    print(textc1)
    
    
ac1=textc1.replace(".","")
textrafc1=ac1.replace("°","")

if(textc1[0]=="-" and textc1[2]=="."):
    a1c1=textrafc1[0:1]
    a2c1=textrafc1[1:3]
    cadenasa1c1=[a1c1,"0",a2c1]
    textrafc1="".join(cadenasa1c1)
    print (textrafc1)  
if(textc1[0]!="+" and textc1[0]!="-"):
    
    cadenasa2c1=["+",textrafc1]
    textrafc1="".join(cadenasa2c1)
    print ("textra2c1=",textrafc1)
    
    if(textrafc1[0]=="+" and textc1[1]=="."):
        a3c1=textrafc1[0]
        a4c1=textrafc1[1:3]
      
        cadenasa3c1=[a3c1,"0",a4c1]
        textrafc1="".join(cadenasa3c1)
        print ("textra3=",textrafc1)  
print(textrafc1)

imagec2=cv2.imread("new_imagec2.png")
newImgc2 = cv2.imread("new_imagec2.png")


grayc2=cv2.cvtColor(newImgc2,cv2.COLOR_BGR2GRAY)

grayc2=cv2.blur(grayc2,(2,2))
thec2=cv2.threshold(grayc2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
cannyc2=cv2.Canny(thec2,0,200)


canny2c2=cv2.dilate(cannyc2,None,iterations=1)

cnts,cnts1=cv2.findContours(cannyc2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image,cnts,171,(0,255,0),2)
textc2=pytesseract.image_to_string(newImgc2,config="--psm 11")
textac2=pytesseract.image_to_string(grayc2,config="--psm 11")
textbc2=pytesseract.image_to_string(cannyc2,config="--psm 11")
textcc2=pytesseract.image_to_string(canny2c2,config="--psm 11")
#print(text.find("02°"))

bc2=len(textc2)
print(bc2)
v1c2=0
v2c2=0
v3c2=0
if(bc2<=5 and v1c2==0):
    v1c2=1
    
if(bc2>=6 and v1c2==0):
    
    print("v1c2",v1c2)
    textc2=textc2[0:7]
    print(textc2)
if(bc2>=5 and v1c2!=1):
    
    print("v2c2",v2c2)
    textc2=textc2[0:6]   
    print(textc2)
if(bc2>=4 and v2c2!=1 and v1c2!=1):
    
    print("v3c2",v3c2)
    textc2=textc2[0:5] 
    print(textc2)
    
    
ac2=textc2.replace(".","")
textrafc2=ac2.replace("°","")

if(textc2[0]=="-" and textc2[2]=="."):
    a1c2=textrafc2[0:1]
    a2c2=textrafc2[1:3]
    cadenasa1c2=[a1c2,"0",a2c2]
    textrafc2="".join(cadenasa1c2)
    print (textrafc2)  
if(textc2[0]!="+" and textc2[0]!="-"):
    
    cadenasa2c2=["+",textrafc2]
    textrafc2="".join(cadenasa2c2)
    print ("textra2c2=",textrafc2)
    
    if(textrafc2[0]=="+" and textc2[1]=="."):
        a3c2=textrafc2[0]
        a4c2=textrafc2[1:3]
      
        cadenasa3c2=[a3c2,"0",a4c2]
        textrafc2="".join(cadenasa3c2)
        print ("textra3c2=",textrafc2)  
print(textrafc2)







d = pytesseract.image_to_data(newImgc1, output_type=Output.DICT)


date_pattern = "0.1"

n_boxes = len(d["text"])
for i in range(n_boxes):
    if float(d["conf"][i]) < 1000:
     	if re.match(date_pattern, d["text"][i]):
 	        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
 	        img = cv2.rectangle(newImgc1, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("img", newImgc1)
cv2.waitKey(0)