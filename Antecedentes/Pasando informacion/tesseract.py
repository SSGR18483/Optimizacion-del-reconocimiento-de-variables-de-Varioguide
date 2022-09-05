import cv2

import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

numero=[]

image=cv2.imread("humana1.png")
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
    if len(approx)<=2.1 and area>1000 and area<3100 :
        #print ("area=",area)
        
        cv2.drawContours(image,[c],0,(0,255,0),2)
        aspect_ratio=float(w)/h
        if aspect_ratio>2.7 and aspect_ratio<2.8:
            placa=image[y:y+h,x:x+w]
            cv2.imwrite("placaA.png", placa)
            text=pytesseract.image_to_string(the,config="--psm 11")
            #print("text=",text[0])
            cv2.imshow("placa",placa)
            cv2.moveWindow("placa",780,10)
cv2.imshow("image",image)
cv2.imshow("canny",canny)
cv2.imshow("threshold",the)
cv2.moveWindow("image",45,10)
cv2.waitKey(0)

texta3=pytesseract.image_to_string(gray,config="--psm 11")
print(texta3.find("Joint 1 Adjustment"))