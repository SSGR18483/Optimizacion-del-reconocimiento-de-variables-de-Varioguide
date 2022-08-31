#UVG
#Santiago Galicia

#importe de librerias
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Obtencion de imagen
img = cv2.imread("c1.png",0)
#Conversion de BGR a RGB para pasarlo a pyplot utilizando color  esto reduce calidad
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Deteccion de bordes Canny
canny= cv2.Canny(img,100,200)
#Configuracion de salida de imagen
titles = ['imagen','Canny']
images = [img, canny]
#Array de imagen
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
