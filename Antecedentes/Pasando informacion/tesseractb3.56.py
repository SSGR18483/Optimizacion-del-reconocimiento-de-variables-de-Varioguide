import cv2




import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

numero=[]

image=cv2.imread("prueba.jpg")
  
image=cv2.resize(image,(2020,1380),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
the=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
gray=cv2.blur(gray,(4,4))
canny=cv2.Canny(gray,0,200)
#canny=cv2.dilate(canny,None,iterations=1)






cnts,cnts1=cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image,cnts,171,(0,255,0),2)


for c in cnts:
    area=cv2.contourArea(c)
    x,y,w,h=cv2.boundingRect(c)
    epsilon=0.1*cv2.arcLength(c,True)
   
    approx=cv2.approxPolyDP(c,epsilon,True)

    if len(approx)<=8 and area>2000 and area<10000:
        print ("area=",area)
        
        cv2.drawContours(image,[c],0,(0,255,0),2)
        aspect_ratio=float(w)/h
        print ("r=",aspect_ratio)
        if aspect_ratio>2.4:
            placab=image[y:y+h,x:x+w]
            cv2.imwrite("placab.png", placab)
            orig_img = cv2.resize(placab, (0,0), fx=3, fy=3,interpolation=cv2.INTER_CUBIC)
            gray=cv2.cvtColor(orig_img,cv2.COLOR_BGR2GRAY)
            the=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            text=pytesseract.image_to_string(the,config="--psm 11")
            cv2.imwrite("placab.png", the)
            print("text=",text[0])
            cv2.imshow("placab",the)
            cv2.moveWindow("placab",780,10)

cv2.imshow("image",image)
cv2.imshow("canny",canny)
cv2.imshow("canny",the)
cv2.moveWindow("image",45,10)
cv2.waitKey(0)