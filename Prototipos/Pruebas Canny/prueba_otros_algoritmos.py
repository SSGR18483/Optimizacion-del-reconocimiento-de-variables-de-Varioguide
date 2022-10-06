#UVG
#Santiago Galicia
#carnet 18483

##Programa de prueba para otros algoritmos y comparacion de detectores de bordes
#Importe de librerias
import cv2
import numpy as np
from matplotlib import pyplot as plt
#Retrieve de la imagen
img = cv2.imread("Switch_01.jfif",cv2.IMREAD_GRAYSCALE)

#Para mostrarlo en la ventana de python se utiliza la siguiente function

def display(im_path):
    dpi =80
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    figsize= width / float(dpi), height / float(dpi)
    fig =plt.figure(figsize=figsize)
    ax= fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    ax.imshow(im_data, cmap='gray')

    plt.show()


#Detector de bordes por medio del laplaciano del gausiano
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
#Detector por medio de Sobel
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombinado = cv2.bitwise_or(sobelX, sobelY)
#Detector por medio de Canny
edgesC = cv2.Canny(img,300,500)
#Despliegue de imagenes
titles = ['Imagen','Laplaciano','Sobel combinado', 'Canny']
images = [img, sobelCombinado, lap, edgesC]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
