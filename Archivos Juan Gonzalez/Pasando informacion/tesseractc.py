# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 00:26:34 2021

@author: joe
"""
import cv2

import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

numero=[]

image=cv2.imread("selfie.png")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
the=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
gray=cv2.blur(gray,(3,3))
canny=cv2.Canny(gray,0,200)
#canny=cv2.dilate(canny,None,iterations=1)

cnts,cnts1=cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image,cnts,171,(0,255,0),2)


for c in cnts:
    area=cv2.contourArea(c)
    x,y,w,h=cv2.boundingRect(c)
    epsilon=0.09*cv2.arcLength(c,True)
   
    approx=cv2.approxPolyDP(c,epsilon,True)
    if len(approx)<=2.5 and area>1900 and area<2100:
        print ("area=",area)
        
        cv2.drawContours(image,[c],0,(0,255,0),2)
        aspect_ratio=float(w)/h
        if aspect_ratio>2.4:
            placac1=image[y:y+h,x:x+w]
            cv2.imwrite("placac1.png", placac1)
            text=pytesseract.image_to_string(the,config="--psm 11")
            print("text=",text[0])
            cv2.imshow("placac1",placac1)
            cv2.moveWindow("placac1",780,10)
    if len(approx)<=2.5 and area>2150 and area<2200:
        print ("area=",area)
        
        cv2.drawContours(image,[c],0,(0,255,0),2)
        aspect_ratio=float(w)/h
        if aspect_ratio>2.4:
            placac2=image[y:y+h,x:x+w]
            cv2.imwrite("placac2.png", placac2)
            text=pytesseract.image_to_string(the,config="--psm 11")
            print("text=",text[0])
            cv2.imshow("placac2",placac2)
            cv2.moveWindow("placac2",780,10)
                 
            
# cv2.imshow("image",image)
# cv2.imshow("canny",canny)
# cv2.moveWindow("image",45,10)
cv2.waitKey(0)
img = cv2.imread('image_test.jpg')
