# Universidad del Valle de Guatemala
# Facultad de Ingenieria
# Departamento de Ingenieria Electronica, Mecatronica y Biomedica
# Santiago Sebastian Galicia Reyes
# TRABAJO DE GRADUACION

# Librerias
import numpy as np
import cv2
# from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import argparse
import time


# PreProcesamiento de imagenes para reconocimiento de caracteres:

#Trabajar abriendo un camara
# Create an object to hold reference to camera video capturing
def obtenercaptura():
    vidcap = cv2.VideoCapture(0)

    # check if connection with camera is successfully
    if vidcap.isOpened():
        ret, frame = vidcap.read()  # capture a frame from live video

        # check whether frame is successfully captured
        if ret:
            print("se pudo capturar el frame")
        # print error if frame capturing was unsuccessful
        else:
            print("Error : no se capturo el frame")

    # print error if the connection with camera is unsuccessful
    else:
        print("Cannot open camera")
    return frame

# abrir una imagen.
# image_file= 'humana1.png' Caso de imagen
image_file='captura1.jpg'
CASO=2
if CASO==1:
    img=obtenercaptura()
elif CASO ==2:
    img=cv2.imread(image_file)
else:
    print("no se pudo xd")

# inversion de colores de imagenes
#tesseract 4.0 no usa esto pero si el 3.0
inverted_image = cv2.bitwise_not(img)

# Reescalamiento
def rescale(image,width, height):
    down_points1=(width,height)
    img_resized=cv2.resize(image,down_points1,interpolation=cv2.INTER_LINEAR)
    return img_resized
imagen_rescalada=rescale(img,350,600)

# Binarizacion
def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_img = grayscale(inverted_image)
gray_img = cv2.bitwise_not(gray_img)
#thresh2 = cv2.threshold(gray_img,120,255,cv2.THRESH_TRUNC)[1]
#thresh2 = cv2.threshold(gray_img,150,180,cv2.THRESH_BINARY_INV)[1]
#thresh2 = cv2.threshold(gray_img,148,180,cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(gray_img,50,70,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#thresh2 =cv2.bitwise_not(thresh2)
# Noise remooval


def noise_removal(image):
    kernel=np.ones((1, 1),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image,cv2.MORPH_CLOSE, kernel)
    image = cv2.GaussianBlur(image,(1,1),0)
    return image

#adelgazando la fuente
def thin(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image,kernel,iterations=1)
    image = cv2.bitwise_not(image)
    return image
#haciendo mas grande la fuente
def thick(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image,kernel,iterations=1)
    image = cv2.bitwise_not(image)
    return image

#Rotation/deskewing
def getSkewAngle(image) -> float:
    newImage = image.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),0)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,5))
    dilate = cv2.dilate(thresh,kernel,iterations=2)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse =True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)
    largestContour=contours[0]
    #print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return (-1.0*angle,newImage)



def rotation(image,angle:float):
    newImage = image.copy()
    (h,w) = newImage.shape[:2]
    center = (w // 2,h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M,(w,h),flags=cv2.INTER_CUBIC, borderMode =cv2.BORDER_REPLICATE)
    return newImage

def deskew(image):
    angle = getSkewAngle(image)[0]
    return rotation(image,-1.0*angle)

# quitar bordes
# esto es super bueno si tenemos bordes incosistentes
def quitar_bordes(image):
    contours, heirarchy = cv2.findContours(image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours, key = lambda x:cv2.contourArea(x))
    cnt= cntSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h,x:x+w]
    return crop


# Agregar bordes que hacen falta
# cuando un caracter esta muy pegado a un borde
def agregar_borde(image):
    color = [0,0,0]
    top, bottom, left, right = [50]*4
    image_extbord = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
    return image_extbord







no_noise = noise_removal(thresh2) # imagen sin ruido
thiner=thin(thresh2) # imagen con letras mas delgadas
thicker =thick(no_noise) #imagen con letras mas gruesas
fixe = deskew(img) # imagen rotada
noboders= quitar_bordes(no_noise) # imagen sin bordes
addborder= agregar_borde(noboders) # imagen con bordes de 50 pts

ocr_result= pytesseract.image_to_string(no_noise)
print(ocr_result)

cv2.imshow("Imagen",no_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()
