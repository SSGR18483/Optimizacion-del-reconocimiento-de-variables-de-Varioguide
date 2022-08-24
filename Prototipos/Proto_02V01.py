#Universidad del Valle de Guatemala
#Facultad de Ingenieria
#Diseño e innovación
#Santiago Galicia - UVG18483
##Programa de prueba para determinar funcionalidad de procesamiento en vivo de video

#librerias
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#programa
cap= cv.VideoCapture(0)

#proteccion
if not cap.isOpened():
    print('no se pudo abrir la camara')
    exit()
while True:
    #Captura frame por frame
    ret, frame = cap.read()

    #proteccion de lectura de frame
    if not ret:
        print('no se pudo recivir frame. ')
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#Gausian blur para suavizar imagen original
    #blur = cv.bilateralFilter(gray,9,75,75)#demasiado violento, obscurese demasiado la imagen
    blur= cv.GaussianBlur(gray,(3,3),0)
#trabajo de procesamiento pruebas: Canny,Sobel,Laplacian
#No.1 Canny
    edgesC = cv.Canny(blur,50,50)
#No.2 Sobel
    sobelX = cv.Sobel(blur, cv.CV_64F, 2, 0)
    sobelY = cv.Sobel(blur, cv.CV_64F, 0, 2)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelCombinado = cv.bitwise_or(sobelX, sobelY)
#No.3 laplaciano
    lap = cv.Laplacian(blur, cv.CV_64F, ksize=3)
    lap = np.uint8(np.absolute(lap))
#CONCATENACION DE IMAGEN PARA MOSTRAR EN OPENCV
    hor1=np.concatenate((gray,sobelCombinado),axis=0)
    hor2=np.concatenate((lap,edgesC),axis=0)
    array=np.concatenate((hor1,hor2),axis=1)

#ventana emergente con camara algoritmo deteccion de ejes
    cv.imshow('Pruebas deteccion ejes',array)
#trabajo de procesamiento pruebas:thresholding

    retval,thresh1 = cv.threshold(gray,120,255,cv.THRESH_BINARY)
    retval,thresh2 = cv.threshold(gray,120,255,cv.THRESH_BINARY_INV)
    retval,thresh3 = cv.threshold(gray,120,255,cv.THRESH_TRUNC)
    retvl,thresh4 = cv.threshold(gray,120,255,cv.THRESH_TOZERO)
    retvl,thresh5 = cv.threshold(gray,120,255,cv.THRESH_TOZERO_INV)
    thresh6 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,199,5)
    thresh7 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,199,5)
#CONCATENACION DE IMAGEN PARA MOSTRAR EN OPENCV
    hor3=np.concatenate((gray,thresh1,thresh2,thresh3),axis=1)
    hor4=np.concatenate((thresh4,thresh5,thresh6,thresh7),axis=1)
    array2=np.concatenate((hor3,hor4),axis=0)
#ventana emergente con pruebas de threshold
    cv.imshow('pruebas thresholding',array2)
#trabajo de procesamiento de transformadores de hough
    if cv.waitKey(1)==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
