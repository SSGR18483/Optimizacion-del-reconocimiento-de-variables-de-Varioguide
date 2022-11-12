# Universidad del Valle de Guatemala
# Facultad de Ingenieria
# Departamento de Ingenieria Electronica, Mecatronica y Biomedica
# Santiago Sebastian Galicia Reyes
# TRABAJO DE GRADUACION

# PROTOTIPE VERSION 03 OPTICAL CHARACTER RECOGNITION

# Librerias
import numpy as np
import cv2
# from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import argparse
import time
import keras_ocr
import pandas as pd
import re
from cv2 import dnn_superres
from skimage import measure
from imutils import contours
from skimage.filters import threshold_multiotsu
import imutils

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
image_file='captura4off.jpg'
#CASO
CASO=2
if CASO==1:
    img=obtenercaptura()
elif CASO ==2:
    img=cv2.imread(image_file)
    imgnp = np.array(Image.open(image_file))
else:
    print("no se pudo xd")


# MANEJO DE DATOS DE LOS SISTEMAS
def Handdle(String):
    txt= String.split(" ")    # s=re.findall(r'\b\d+\b',String)
    if 'Joint' in txt:
        if '1' in txt:
            Joint = 1
        elif '2' in txt:
            Joint = 2
        elif '3' in txt:
            Joint = 3
        aftermant = 'Se leyó adecuadamente'
    if 'Joint' not in txt:
        aftermant = 'No se pudo leer correctamente'
    return  Joint,aftermant


# inversion de colores de imagenes
#tesseract 4.0 no usa esto pero si el 3.0
def inversion(img):
    inverted_image = cv2.bitwise_not(img)
    return inverted_image
#Rescale
def dnnrescale(image,n):
    sr = dnn_superres.DnnSuperResImpl_create()
    imagednn =image
    path = "FSRCNN_x2.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn",n)
    img_dnresized = sr.upsample(imagednn)
    return img_dnresized

# Resize
def rescale(image,width, height):
    down_points1=(width,height)
    img_resized=cv2.resize(image,down_points1,interpolation=cv2.INTER_LINEAR)
    return img_resized
imagen_rescalada=rescale(img,350,600)

# Binarización
def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Thresholding
def umbral(image):
    # threshim = cv2.threshold(gray_img,120,255,cv2.THRESH_TRUNC)[1]
    # threshim = cv2.threshold(gray_img,150,180,cv2.THRESH_BINARY_INV)[1]
    # threshim = cv2.threshold(gray_img,148,180,cv2.THRESH_BINARY)[1]
    threshim = cv2.threshold(image, 107,510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # threshim =cv2.bitwise_not(threshim)
    return threshim


# Noise remooval
def noise_removal(image):
    kernel = np.ones((3, 3),np.uint8)
    image = cv2.erode(image, kernel, iterations=2)
    kernel=np.ones((2, 2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=3)
    image = cv2.morphologyEx(image,cv2.MORPH_CLOSE, kernel)
    return image

#adelgazando la fuente
def thin(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((3,3),np.uint8)
    image = cv2.erode(image,kernel,iterations=2)
    image = cv2.bitwise_not(image)
    return image
#haciendo mas grande la fuente
def thick(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image,kernel,iterations=3)
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

def save_img(image,filename):
    status = cv2.imwrite(filename,image)
    return status

def crop_img(image):#,x,y):#imgnp
    fig0=image[ 350:800,1050:1550,:]
    return fig0

def Blurred(image):
    image = cv2.GaussianBlur(image,(3,3),0)
    return image



#img es la imagen original
inv_img= inversion(img)
blur_img= Blurred(inv_img)
res_img= dnnrescale(img,2)#2x 1080x1920
gray_img = grayscale(res_img)
umb_img = umbral(gray_img)
nonoise_img = noise_removal(umb_img) # imagen sin ruido
thin_img = thin(nonoise_img) # imagen con letras mas delgadas
thick_img = thick(nonoise_img) #imagen con letras mas gruesas
reduced = rescale(thick_img,2560,1440)
#rot_img = deskew(nonoise_img) # imagen rotada
#nobor_img = quitar_bordes(rot_img) # imagen sin bordes
#bor_img= agregar_borde(rot_img) # imagen con bordes de 50 pts
cv2. imshow('umb',reduced)
cv2.waitKey(0)
cv2.destroyAllWindows()


def glare_mask(image): #imagen gris
    #blurred = cv2.GaussianBlur( image, (1,1), 0 )
    _,thresh_img = cv2.threshold( image, 180, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode( thresh_img, None, iterations=2 )
    thresh_img  = cv2.dilate( thresh_img, None,iterations=4 )
    #desarrollar un analisis de componentes conectados en la imagen umbralizada y se hace un la mascara de largos componentes
    labels = measure.label( thresh_img, background=0)
    mask = np.zeros( thresh_img.shape, dtype="uint8" )
    # loop over the unique components
    for label in np.unique( labels ):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero( labelMask )
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add( mask, labelMask )
    return mask



#probar antiglare
# blur_img= Blurred(img)
# res_img= rescale(blur_img,width=1920,height=1080)#2x 1080x1920
# gray_img = grayscale(res_img)
# glaresed = cv2.inpaint(gray_img , glare_mask(gray_img), 3,cv2.INPAINT_NS)
# nonoise_img = noise_removal(glaresed) # imagen sin ruido
# umb_img = umbral(nonoise_img)
# thin_img = thin(nonoise_img) # imagen con letras mas delgadas
# thick_img = thick(nonoise_img) #imagen con letras mas gruesas


# cv2. imshow('glared',umb_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
