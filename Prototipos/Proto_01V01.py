#UNIVERSIDAD DEL VALLE DE GUATEMALA
#FACULTAD DE INGENIERIA
#TRABAJO DE GRADUACION
#SANTIAGO GALICIA
#CARNET UVG-18483
#PRIMER PROTOTIPO DE PRUEBA
##################################################################################
#Importar los paquetes necesarios
import numpy as np
import pickle
import math
import cv2 as cv
from matplotlib import pyplot as plt

##1ER ALGORITMO: DETECCION DE BORDES CANNY, SOBEL Y LAPLACIANO DEL GAUSIANO
img = cv.imread("GATO.jpg",cv.IMREAD_GRAYSCALE)
lap = cv.Laplacian(img, cv.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
#Detector por medio de Sobel
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)
sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombinado = cv.bitwise_or(sobelX, sobelY)
#Detector por medio de Canny
edgesC = cv.Canny(img,300,500)
#Despliegue de imagenes
titles = ['Imagen','Laplaciano','Sobel combinado', 'Canny']
images = [img, sobelCombinado, lap, edgesC]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##2DO ALGORITMO: THRESHOLDING
image1 = cv.imread('LIBRO.jpeg')
imagen = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
ret,thresh1 = cv.threshold(imagen,120,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(imagen,120,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(imagen,120,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(imagen,120,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(imagen,120,255,cv.THRESH_TOZERO_INV)
thresh6 = cv.adaptiveThreshold(imagen,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,199,5)
thresh7 = cv.adaptiveThreshold(imagen,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,199,5)
#cv2.imshow('Binary Threshold', thresh1)
#cv2.imshow('Binary Threshold Inverted', thresh2)
#cv2.imshow('Truncated Threshold', thresh3)
#cv2.imshow('Set to 0', thresh4)
#cv2.imshow('Set to 0 Inverted', thresh5)
#cv.imshow('Adaptative Mean', thresh6)
#cv.imshow('Adaptative Gaussian', thresh7)
titulos_thres = ['Imagen Original','Threshold binario','Threshold binario invertido',
                 'Threshold truncado' ,'Threshold puesto a 0','Threshold puesto a 0 invertido',
                 'Threshold adaptativo','Threshold adaptativo gausiano']
imagenes_thres = [image1,thresh1,thresh2,thresh3,thresh4,thresh5,thresh6,thresh7]

for k in range(8):
    plt.subplot(2, 4, k+1)
    plt.imshow(imagenes_thres[k], 'gray')
    plt.title(titulos_thres[k])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# De-allocate any associated memory usage 
#if cv2.waitKey(0) & 0xff == 27:
    #cv2.destroyAllWindows()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##3ER ALGORITMO: TRANSFORMADORES DE HOUGH
#Transformada de linea de HOGH
src_img = cv.imread('dots.png',0)
color_img = cv.cvtColor(src_img,cv.COLOR_GRAY2BGR)

edgesC2 = cv.Canny(color_img,0,10)
cdst = cv.cvtColor(edgesC2, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)
lines = cv  .HoughLines(edgesC2, 1, np.pi / 180, 150, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
#Transformada de circulo de HOUGH
#src_img = cv.imread('dots.png',0)
#color_img = cv.cvtColor(src_img,cv.COLOR_GRAY2BGR)
circles_img = cv.HoughCircles(src_img,cv.HOUGH_GRADIENT,1,2,param1=100,param2=30,minRadius=0,maxRadius=0)
circles_img = np.uint16(np.around(circles_img))
for i in circles_img[0,:]:
    cv.circle(color_img,(i[0],i[1]),i[2],(0,255,0),2)
    cv.circle(color_img,(i[0],i[1]),2,(0,0,255),3)
plt.subplot(2,1,1)
plt.imshow(cdst)
plt.title('Transformada de linea de HOUGH')
plt.subplot(2,1,2)
plt.imshow(color_img)
plt.title('Transformada de circulo de HOUGH')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
##4TO ALGORITMO: 
