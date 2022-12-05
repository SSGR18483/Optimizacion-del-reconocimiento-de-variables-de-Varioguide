#
#     __  ___    ________
#    / / / / |  / / ____/
#   / / / /| | / / / __
#  / /_/ / | |/ / /_/ /
#  \____/  |___/\____/
#
#  Sistema ocr HUMANA
#

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
import glob
import tensorflow as tf
import os
from skimage.filters import threshold_multiotsu
from skimage import measure
from imutils import contours


# PreProcesamiento de imagenes para reconocimiento de caracteres:

#Trabajar abriendo un camara
# Create an object to hold reference to camera video capturing
def obtenercaptura():
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    vidcap.set(28, 30)
    widthc = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    heightc = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(widthc, heightc)
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


# MANEJO DE DATOS DE LOS SISTEMAS
def Handdle(String):
    if String.find('Joint 1') >=0 or String.find('int 1') >=0:
        Joint = 1;
        aftermath = 'Se leyó adecuadamente';
    elif String.find('Joint 2') >=0 or String.find('int 2')>=0:
        Joint = 2;
        aftermath = 'Se leyó adecuadamente';
    elif String.find('Joint 3') >=0 or String.find('int 3')>=0:
        Joint = 3;
        aftermath = 'Se leyó adecuadamente';
    else:
        Joint = 0;
        aftermath = 'No se pudo leer adecuadamente, intente de nuevo'
    return Joint,aftermath

def signohanddle(string):
    if string.find(' - ') >=0 or string.find('-') >=0:
        signo = -1
    else:
        signo = 1
    return signo


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
    fig0=image[ cy-40:cy+40,cx-95:cx+90,:]
    return fig0

def Blurred(image):
    image = cv2.GaussianBlur(image,(5,5),3)
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
    # guardar =picture
    blurred = cv2.GaussianBlur(imagen, (5, 5), 0)
    #lower = np.array([39, 40, 38])
    lower = np.array([38,38,36])# en caso sea ultimas fotos
    # lower = np.array([45, 50, 49])  # en caso sea live
    #upper = np.array([55, 57, 75])
    upper = np.array([73,73,72]) #en caso sea ultimas fotos
    # upper = np.array([56, 63, 72])  # en caso sea live
    mask = cv2.inRange(blurred, lower, upper)
    contours , _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            # cv2.drawContours(guardar, contour, -1, (0, 255, 0), 3)
            # estado = cv2.imwrite('Primercontorno.jpg', cv2.cvtColor(guardar,cv2.COLOR_RGB2BGR))
            M = cv2.moments(contour)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            #cv2.circle(imagen,(cx,cy),7,(255,255,255),-1)
    return imagen,cx,cy

def recorte_inicial(image,cx,cy):#,x,y):#imgnp cx es 960 y cy es 540
    # fig0=image[ cy-505:cy+505,cx-665:cx+640,:]
    fig0 = image[cy - 540:cy + 540, cx - 680:cx + 640, :]
    # fig0 = image[cy - 485:cy + 485, cx - 660:cx + 631, :]
    # fig0 = image[cy - 456:cy + 456, cx - 577:cx + 577, :]
    return fig0

def contorno_numeros(corte): # entra imagen normal y sale imagen normal con contornos de numeros.# procurar que sea la imagen cortada.
    roi = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi,(3,3),0)
    No_dig = 1
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.threshold(blur, 107, 510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel = np.ones((3, 4), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        try:
            if (w*h > 240) & (w*h < 600) :
                # cv2.rectangle(corteguardar, (x-(2), y-(2)), (x + w+(2), y + h+(2)), color=(0, 255, 0), thickness=2)
                # estado = cv2.imwrite('contornosnum.jpg', corteguardar)
                fig0 = roi[y-2:y+h+2, x-3:x+w+3]
                fig0 = cv2.cvtColor(fig0, cv2.COLOR_GRAY2RGB)
                # plt.imshow(fig0, cmap='gray')
                # plt.title('Example', fontweight="bold")
                # plt.show()
                # digit = thresh[y:y + h, x:x + w]
                # resized_digit = cv2.resize(digit, (18, 18))
                # plt.imshow(fig0,cmap='gray')
                # plt.title('Example', fontweight="bold")
                # plt.show()
                filename = f"digit{No_dig}.png"
                status = cv2.imwrite(filename, fig0)
                No_dig += 1
        except:
            print(f"Error al procesar No. {No_dig}\n")
    return (corte)


def keras_dr(ad1):#funcion que recibe la direccion de la imagen guardada o una variable y entrega un string con los digitos analizados de keras.
    pipeline = keras_ocr.pipeline.Pipeline()
    image = [keras_ocr.tools.read(ad1)]
    imagenkeras = np.array(image)
    resultado= pipeline.recognize(imagenkeras)
    #a = pd.DataFrame(resultado[0], columns=['text','bbox'])
    a = resultado[0][0]
    sfj = a[0]
    return sfj
#ejemplo keras:
#img_ker='cutted.jpg'
# sfj= keras_dr('cutted.jpg')
# print(sfj)
# print(sfj[0:2]) #[0] es  para el primer digito. [1] es para el decimal.

def no_proces(imagen): #funcion que recibe imagen y que la regresa en blanco y negro con dimensiones de 28x28
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    (thresh, imagen) = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    imagen = cv2.resize(imagen, (28, 28), interpolation=cv2.INTER_LINEAR)
    imagen = np.array([imagen])
    return imagen

def graf_DNN(history,epochs): #funcion que recibe un model fit con epochs y grafica el desempeño del modelo
    #recomendable utilizar un model.fit con datos de entrenamiento, epochs y los datos de validacion
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy de entrenamiento vs validacion')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Perdida de entrenamiento y validacion')
    plt.show()
    return


def processsing(img):
    CUTGR = grayscale(img)
    CUTBLR = Blurred(CUTGR)
    CUTUM = umbral(CUTBLR)
    CUTINV = inversion(CUTUM)
    return CUTINV
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#                                                                                                       ALGORITMO PRINCIPAL
#...............................................................................................................................................................................................................................

# abrir una imagen.
# image_file= 'humana1.png' Caso de imagen





image_file='J1[20.6].jpg'




#CASO
CASO=2
if CASO==1:
    img=obtenercaptura()
elif CASO ==2:
    img=cv2.imread(image_file)
    imgnp = np.array(Image.open(image_file))
else:
    print("no se pudo xd")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img,cmap=plt.cm.binary)
# plt.show()
# estado = save_img(img,'normal.jpg')

#img es la imagen original
Corte1 = recorte_inicial(img,960,540)
# status = cv2.imwrite('corteini.jpg',cv2.cvtColor(Corte1, cv2.COLOR_RGB2BGR))
imagenf,cx,cy=dibujo_contornos(Corte1)
plt.imshow(imagenf)
trim = crop_img(imagenf,cx,cy)
estado = save_img(trim,'cutted.jpg')

image_archivo='cutted.jpg'
img_color_cut = cv2.imread(image_archivo)
figurita = contorno_numeros(img_color_cut)

# Reconocimiento de Caracteres por medio de Tesseract
res_img= dnnrescale(Corte1,2)#width=3840,height=2160)#2x 1080x1920
gray_img = grayscale(res_img)
blur_img = Blurred(gray_img)
umb_img = umbral(blur_img)
nonoise_img = noise_removal(umb_img) # imagen sin ruido
thin_img = thin(nonoise_img) # imagen con letras mas delgadas
thick_img = thick(nonoise_img) #imagen con letras mas gruesas
ocr_result= pytesseract.image_to_string(nonoise_img)
estado= save_img(umb_img,'processed.jpg')

# print(ocr_result)
Joints,mensajito = Handdle(str(ocr_result))
if mensajito == 'No se pudo leer adecuadamente, intente de nuevo':
    try:
        ocr_result = pytesseract.image_to_string(gray_img)
        print(ocr_result)
    except:
        print("no se pudo arreglar dvd")
Joints,mensajito = Handdle(str(ocr_result))
procesosingo=processsing(img_color_cut)
ocr_resultsign=pytesseract.image_to_string(procesosingo)
print(mensajito)
print(Joints)
print("-----------------")
if Joints == 1 or Joints == 3 or Joints == 0:
    try:
        texto = ocr_resultsign[0:5]
        numeropytess=float(texto.replace('°','').replace('º',''))
    except:
        texto=ocr_resultsign[0:4]
        numeropytess = float(texto.replace('°', '').replace('º',''))
elif Joints == 2:
    numeropytess=float(ocr_resultsign.replace('º','').replace('º',''))
    numeropytess2=float(ocr_resultsign.replace('mm','').replace('nm','').replace('mn',''))
else:
    print("xd")
# print(float(ocr_resultsign))
signo = signohanddle(str(ocr_resultsign))
print("-----------------")
print(signo)
print(numeropytess)

mnist = tf.keras.datasets.mnist
#importe de set de datos y proceso de datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train / 255
X_test = x_test / 255
X_train = X_train.reshape(-1,28,28,1)    #training set
X_test = X_test.reshape(-1,28,28,1)

# model= tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(120, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# epochs=22
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history=model.fit(X_train,y_train, epochs=epochs,validation_data=(X_test,y_test))
# model.save('cnn.model')

model = tf.keras.models.load_model('cnn.model')
loss, accuracy = model.evaluate(X_test, y_test)
# graf_DNN(history,epochs)
digit1 = 0;digit2=0;digit0=0;
# model = tf.keras.models.load_model('cnn.model')
image_no=1
while os.path.isfile(f"digit{image_no}.png"):
    try:
        path = f"digit{image_no}.png"
        img = cv2.imread(path)
        img = no_proces(img)
        # plt.imshow(img[0],cmap=plt.cm.binary)
        # plt.show()
        prediction = model.predict(img)
        # print(f"el numero es probablemente un {np.argmax(prediction)}")
        if image_no ==1:
            digit2 = np.argmax(prediction)
        elif image_no ==2:
            digit1 =np.argmax(prediction)
        elif image_no ==3:
            digit0 = np.argmax(prediction)
        else:
            print("Error")
    except:
        print("Error!")
    finally:
        image_no +=1

digitos= float(digit0*10+digit1+(digit2/10))
if digitos == numeropytess:
    print("````````````````````````````````````````````````````````````````````````")
    print(f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {signo*digitos}")
    print("........................................................................")
elif digitos != numeropytess:
    print("````````````````````````````````````````````````````````````````````````")
    print(f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {numeropytess}")
    print("........................................................................")
else:
    print("no funciono bien ")











