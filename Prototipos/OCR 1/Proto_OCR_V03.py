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

# inversion de colores de imagenes
#tesseract 4.0 no usa esto pero si el 3.0
def inversion(img):
    inverted_image = cv2.bitwise_not(img)
    return inverted_image

# Reescalamiento
def rescale(image,width, height):
    down_points1=(width,height)
    img_resized=cv2.resize(image,down_points1,interpolation=cv2.INTER_LINEAR)
    return img_resized
imagen_rescalada=rescale(img,350,600)

# Binarizacion
def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def umbral(image):
    # threshim = cv2.threshold(gray_img,120,255,cv2.THRESH_TRUNC)[1]
    # threshim = cv2.threshold(gray_img,150,180,cv2.THRESH_BINARY_INV)[1]
    # threshim = cv2.threshold(gray_img,148,180,cv2.THRESH_BINARY)[1]
    threshim = cv2.threshold(image, 107,510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # threshim =cv2.bitwise_not(threshim)
    return threshim


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

def save_img(image,filename):
    status = cv2.imwrite(filename,image)
    return status

def crop_img(image):#,x,y):#imgnp
    fig0=image[ 250:800,1000:1600,:]
    return fig0

#img es la imagen original
inv_img= inversion(img)
res_img= rescale(inv_img,width=3840,height=2160)#2x 1080x1920
gray_img = grayscale(res_img)
umb_img = umbral(gray_img)
nonoise_img = noise_removal(umb_img) # imagen sin ruido
thin_img = thin(nonoise_img) # imagen con letras mas delgadas
thick_img = thick(nonoise_img) #imagen con letras mas gruesas
#rot_img = deskew(nonoise_img) # imagen rotada
#nobor_img = quitar_bordes(rot_img) # imagen sin bordes
#bor_img= agregar_borde(rot_img) # imagen con bordes de 50 pts
estado= save_img(nonoise_img,'processed.jpg')
trim = crop_img(imgnp)
plt.imshow(trim)
estado = save_img(trim,'cutted.jpg')


pipeline = keras_ocr.pipeline.Pipeline()
image_keras=[keras_ocr.tools.read(imga) for imga in ['processed.jpg', 'cutted.jpg']]
imageneskeras=np.array(image_keras)
print(len(imageneskeras))
results = pipeline.recognize(imageneskeras)
print(pd.DataFrame(results[0], columns=['text', 'bbox']))
fig, axs = plt.subplots(nrows=len(imageneskeras), figsize = (20,20))
for ax, image,predictions in zip(axs,imageneskeras,results):
    keras_ocr.tools.drawAnnotations(image=image,
                                    predictions=predictions,
                                    ax=ax)
plt.show()
print(pd.DataFrame(results[1], columns=['text', 'bbox']))


#reconocimiento de Caracteres por medio de Tesseract
ocr_result= pytesseract.image_to_string(nonoise_img)
res_img= rescale(img,width=3840,height=2160)#2x 1080x1920
gray_img = grayscale(res_img)
umb_img = umbral(gray_img)
nonoise_img = noise_removal(umb_img) # imagen sin ruido
thin_img = thin(nonoise_img) # imagen con letras mas delgadas
thick_img = thick(nonoise_img) #imagen con letras mas gruesas
ocr_result2= pytesseract.image_to_string(nonoise_img)
print(ocr_result)
print(ocr_result2)


# EN IMAGEN RECORTADA
res_trim= rescale(trim,width=1080,height=1920)#2x 1080x1920
gray_trim = grayscale(res_trim)
umb_trim = umbral(gray_trim)
nonoise_trim = noise_removal(umb_trim) # imagen sin ruido
ocr_result3= pytesseract.image_to_string(nonoise_trim)
print(ocr_result3)
# cv2.imshow("Imagen",no_noise)
# cv2.waitKey(0)
# cv2.destroyAllWindows()