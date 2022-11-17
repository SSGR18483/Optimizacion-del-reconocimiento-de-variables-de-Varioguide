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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
#imagen_rescalada=rescale(img,350,600)

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
    kernel=np.ones((1, 1),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
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
    image = cv2.dilate(image,kernel,iterations=2)
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

def crop_img(image,cx,cy):#,x,y):#imgnp
    fig0=image[ cy-40:cy+40,cx-100:cx+105,:]
    return fig0

def Blurred(image):
    image = cv2.GaussianBlur(image,(5,5),0)
    return image

def mask_manual(img,modo): #https://www.youtube.com/watch?v=YRb48EUk6Dk /// https://notebook.community/ricklon/opencvraspberrypi/notebook/openCV%20color%20detection
    pic=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if modo == 1:
        mask1 = cv2.inRange(pic, (41, 41, 35), (49, 51, 76))
        mask2 = cv2.inRange(pic, (36, 36, 36), (55, 58, 76))
        mask = cv2.bitwise_or(mask1, mask2)
    elif modo == 2:
        mask = cv2.inRange(pic,(39,40,38),(55,57,75))
    elif modo == 3:
        mask = cv2.inRange(pic,(38,38,36),(73,73,72))
    return mask
def adaptUMB(img):
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return dst

def dibujo_contornos(picture):
    imagen = picture
    blurred = cv2.GaussianBlur(imagen, (5, 5), 0)
    lower = np.array([39, 40, 38])
    #lower = np.array([38,38,36]) en caso sea ultimas fotos
    upper = np.array([55, 57, 75])
    #upper = np.array([73,73,72]) en caso sea ultimas fotos
    mask = cv2.inRange(blurred, lower, upper)
    contours , _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            cv2.drawContours(imagen, contour, -1, (0, 255, 0), 3)
            M = cv2.moments(contour)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            #cv2.circle(imagen,(cx,cy),7,(255,255,255),-1)
    return imagen,cx,cy

def recorte_inicial(image,cx,cy):#,x,y):#imgnp cx es 960 y cy es 540
    fig0=image[ cy-505:cy+505,cx-660:cx+640,:]
    return fig0

#img es la imagen original
imagenf,cx,cy=dibujo_contornos(hsv)
inv_img= inversion(img)
blur_img= Blurred(inv_img)
res_img= dnnrescale(img,2)#2x 1080x1920
gray_img = grayscale(res_img)
umb_img = umbral(gray_img)
nonoise_img = noise_removal(umb_img) # imagen sin ruido
thin_img = thin(nonoise_img) # imagen con letras mas delgadas
thick_img = thick(nonoise_img) #imagen con letras mas gruesas
#rot_img = deskew(nonoise_img) # imagen rotada
#nobor_img = quitar_bordes(rot_img) # imagen sin bordes
#bor_img= agregar_borde(rot_img) # imagen con bordes de 50 pts
estado= save_img(umb_img,'processed.jpg')
trim = crop_img(imgnp,cx,cy)
estado = save_img(trim,'cutted.jpg')




# Reconocimiento de Caracteres por medio de Tesseract

res_img= dnnrescale(img,2)#width=3840,height=2160)#2x 1080x1920
gray_img = grayscale(res_img)
blur_img = Blurred(gray_img)
umb_img = umbral(blur_img)
nonoise_img = noise_removal(umb_img) # imagen sin ruido
thin_img = thin(nonoise_img) # imagen con letras mas delgadas
thick_img = thick(nonoise_img) #imagen con letras mas gruesas
ocr_result= pytesseract.image_to_string(nonoise_img)
print(ocr_result)


# EN IMAGEN RECORTADA
res_trim= dnnrescale(trim,2)#2x 900*600
gray_trim = grayscale(res_trim)
blur_trim = Blurred(gray_trim)
umb_trim = umbral(gray_trim)
nonoise_trim = noise_removal(umb_trim) # imagen sin ruido
thick_trim = thick(nonoise_trim)
thick_trim2 = thick(thick_trim)
thick_trim = cv2.bitwise_not(thick_trim2)
ocr_result3= pytesseract.image_to_string(thick_trim)#, config='digits') # con configuracion de digits no funciona bien, no lee nada.
print('Digitos detectados:')
print(ocr_result3)


# KERAS OCR # por el momento no me es util
# pipeline = keras_ocr.pipeline.Pipeline()
# image_keras=[keras_ocr.tools.read(imga) for imga in ['processed.jpg', 'cutted.jpg']]
# imageneskeras=np.array(image_keras)
# print(len(imageneskeras))
# results = pipeline.recognize(imageneskeras)
# print(pd.DataFrame(results[0], columns=['text', 'bbox']))
# fig, axs = plt.subplots(nrows=len(imageneskeras), figsize = (20,20))
# for ax, image,predictions in zip(axs,imageneskeras,results):
#    keras_ocr.tools.drawAnnotations(image=image,
#                                    predictions=predictions,
#                                    ax=ax)
# plt.show()
# print(pd.DataFrame(results[1], columns=['text', 'bbox']))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Opciones para resolver la busqueda de punto : %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1. aplicar mascara para buscar el punto a ver si esta en misma posicion siempre. ya sea la misma posicion en un lugar de la imagen o en la misma posicion del ovalo de numeros
# 2. En otro caso buscar el circulo del punto y separar ambos numeros y construir el numero en post procesing
# 3. forma de busqueda pasada con Canny y transformadas.

# 1. Mascara:
# el punto en la imagen de la junta 1 esta en X: 1293 Y:500   ; en la junta 2 esta en X1:1302  Y1:288 y X2:1278 Y2:506  en la junta 3 esta en X:1289 Y:506
# 2.Transformada de Hough.
#rowsc = blur_trim.shape[0]
#circles = cv2.HoughCircles(blur_trim, cv2.HOUGH_GRADIENT, 1, rowsc / 4,param1=200, param2=25,minRadius=0, maxRadius=10)
#if circles is not None:
#    circles = np.uint16(np.around(circles))
#    for i in circles[0, :]:
#        center = (i[0], i[1])
#        # circle center
#        cv2.circle(res_trim, center, 1, (0, 100, 100), 3)
#        # circle outline
#        radius = i[2]
#        cv2.circle(res_trim, center, radius, (255, 150, 255), 3)

# 3. Busqueda con Canny y procesamiento.
edgesC = cv2.Canny(thick_trim,180,150, apertureSize=5)
ocr_result4= pytesseract.image_to_string(edgesC, config='digits')

print('Digitos detectados con Canny:')
print(ocr_result4)
print(Handdle(ocr_result))

estado2= save_img(thick_trim,'procesadocann.jpg')
pipeline = keras_ocr.pipeline.Pipeline()
image_keras=[keras_ocr.tools.read(imga) for imga in ['procesadocann.jpg', 'cutted.jpg']]
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PAGINAS QUE PUEDEN SER UTILES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Upscale
#https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066
#tutorial tkinter
#https://www.geeksforgeeks.org/python-tkinter-tutorial/#introduction
#image procesing ocr
#https://github.com/schollz/python-ocr/blob/master/process_image.py
#tesseract options
#https://muthu.co/all-tesseract-ocr-options/
#entrenamiento de keras ocr
#https://colab.research.google.com/drive/1PxxXyH3XaBoTgxKIoC9dKIRo4wUo-QDg#scrollTo=I7SF5VeoLulc
#SVHN digit detector
#https://github.com/penny4860/SVHN-deep-digit-detector
#Digit detector with Mnist
#https://towardsdatascience.com/build-a-multi-digit-detector-with-keras-and-opencv-b97e3cd3b37
#Train custom pipeline with keras ocr
#https://colab.research.google.com/drive/19dGKong-LraUG3wYlJuPCquemJ13NN8R
#GLARE
#https://pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
#https://stackgod.blogspot.com/2021/06/glare-removal-with-inpaintingopencv.html
#https://rcvaram.medium.com/glare-removal-with-inpainting-opencv-python-95355aa2aa52
#paper con image segmentation
#https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Atienza_Data_Augmentation_for_Scene_Text_Recognition_ICCVW_2021_paper.pdf
#https://splunktool.com/python-opencv-ocr-image-segmentation
#paper sobre calidad adecuada de imagenes para ocr
#https://groups.google.com/g/tesseract-ocr/c/Wdh_JJwnw94/m/24JHDYQbBQAJ
#mejorar calidad ocr
#https://tesseract-ocr.github.io/tessdoc/ImproveQuality
#videos OCR
#https://www.youtube.com/watch?v=4uWp6dS6_G4&list=PL2VXyKi-KpYuTAZz__9KVl1jQz74bDG7i
#https://www.youtube.com/watch?v=PY_N1XdFp4w
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#MNIST DIGIT DETECTOR
#https://towardsdatascience.com/build-a-multi-digit-detector-with-keras-and-opencv-b97e3cd3b37
#Digit Detector
#https://www.youtube.com/watch?v=PHl8NJKpauc
#Color spaces
#https://programmingdesignsystems.com/color/color-models-and-color-spaces/index.html

#imagenes  de HUMANA
#https://www.karger.com/Article/Pdf/510007