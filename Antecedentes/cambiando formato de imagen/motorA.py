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

image=cv2.imread("new_imageA.png")
newImg = cv2.imread("new_imageA.png")


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
#print(text.find("02°"))

b=len(text)
print(b)
v1=0
v2=0
v3=0
if(b<=5 and v1==0):
    v1=1
    
if(b>=6 and v1==0):
    
    print("v1",v1)
    text=text[0:7]
    print(text)
if(b>=5 and v1!=1):
    
    print("v2",v2)
    text=text[0:6]   
    print(text)
if(b>=4 and v2!=1 and v1!=1):
    
    print("v3",v3)
    text=text[0:5] 
    print(text)
    
  
a=text.replace(".","")
textraf=a.replace("°","")

if(text[0]=="-" and text[2]=="."):
    a1=textraf[0:1]
    a2=textraf[1:3]
    cadenasa1=[a1,"0",a2]
    textraf="".join(cadenasa1)
    print (textraf)  
if(text[0]!="+" and text[0]!="-"):
    
    cadenasa2=["+",textraf]
    textraf="".join(cadenasa2)
    print ("textra2=",textraf)
    
    if(textraf[0]=="+" and text[1]=="."):
        a3=textraf[0]
        a4=textraf[1:3]
      
        cadenasa3=[a3,"0",a4]
        textraf="".join(cadenasa3)
        print ("textra3=",textraf)  
print(textraf)
cadenasa4=["f","S",textraf,"Z"]
textrafb="".join(cadenasa4)        
print(textrafb)
                    
                    
                    # string to bytes using bytes()
textrafb1 = bytes(textrafb, encoding='utf-8')
print(textrafb1)


# cv2.imshow("image",image)
# cv2.imshow("re",newImg)
# cv2.imshow("gray",gray)
# cv2.imshow("canny",canny)
# cv2.imshow("the",the)

# cv2.moveWindow("image",45,10)
# cv2.waitKey(0)


d = pytesseract.image_to_data(newImg, output_type=Output.DICT)
keys = list(d.keys())

date_pattern = "0.2"

# n_boxes = len(d["text"])
# for i in range(n_boxes):
#     if float(d["conf"][i]) > 60:
#     	if re.match(date_pattern, d["text"][i]):
# 	        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
# 	        img = cv2.rectangle(newImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow("img", img)
# cv2.waitKey(0)