#Universidad del Valle de Guatemala
#Facultad de Ingenieria
#Diseño e innovación
#Santiago Galicia - UVG18483
##Programa de prueba para determinar funcionalidad de procesamiento

#IMPORTE DE LIBRERIAS NECESARIAS
import numpy as np
import cv2 as cv
import os
#INICIO DE PROGRAMA DE PRUEBAS PARA CAMARAS
#cap= cv.VideoCapture(1)
pathimagenes='prueba4802.jpg'
frame= cv.imread(pathimagenes,cv.IMREAD_COLOR)
down_width1=1920
down_height1=1080
down_points1=(down_width1,down_height1)
frame=cv.resize(frame,down_points1,interpolation=cv.INTER_LINEAR)
cont=0
#direccion=r'C:\Users\galic\Documents\Diseño\GIT\Trabajo-de-Graduaci-n-SG18483\Prototipos\Camera score testing'
direccion=r'D:\Documentos\UVG\QUINTO AÑO\Segundo Semestre\Diseño e innovación\GIT\Optimizacion-del-reconocimiento-de-variables-de-Varioguide\Prototipos\Camera score testing'
#PROTECCION CONTRA FALTA DE LECTURA
#if not cap.isOpened():
    #print('no se pudo abrir la camara')
    #exit()
#Captura frame por frame
##ret, frame = cap.read()

#proteccion de lectura de frame

   # break
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#PROCESAMIENTO DE SUAVIZADO DE IMAGEN
#blur = cv.bilateralFilter(gray,9,75,75)#demasiado violento, obscurese demasiado la imagen
#Gausian blur para suavizar imagen original
blur= cv.GaussianBlur(gray,(3,3),0)
#PROCESAMIENTO A: PRUEBA DE DETECTORES DE EJES
#detector No.1 Canny
edgesC = cv.Canny(blur,150,10, apertureSize=3)
#detector No.2 Sobel
sobelX = cv.Sobel(blur, cv.CV_64F, 2, 0)
sobelY = cv.Sobel(blur, cv.CV_64F, 0, 2)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombinado = cv.bitwise_or(sobelX, sobelY)
#detector No.3 laplaciano
lap = cv.Laplacian(blur, cv.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
#Concatenacion de los videos en una sola ventana
hor1=np.concatenate((gray,sobelCombinado),axis=0)
hor2=np.concatenate((lap,edgesC),axis=0)
array=np.concatenate((hor1,hor2),axis=1)
down_width2=1920
down_height2=1080
down_points2=(down_width2,down_height2)
resize2=cv.resize(array,down_points2,interpolation=cv.INTER_LINEAR)
#ventana emergente con camara algoritmo deteccion de ejes
##    cv.imshow('Pruebas deteccion ejes',array)
#PROCESAMIENTO B: PRUEBAS DE THRESHOLDING

thresh1 = cv.threshold(gray,120,255,cv.THRESH_BINARY)[1]
thresh2 = cv.threshold(gray,120,255,cv.THRESH_BINARY_INV)[1]
thresh3 = cv.threshold(gray,120,255,cv.THRESH_TRUNC)[1]
thresh4 = cv.threshold(gray,120,255,cv.THRESH_TOZERO)[1]
thresh5 = cv.threshold(gray,120,255,cv.THRESH_TOZERO_INV)[1]
thresh6 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,199,5)
thresh7 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,199,5)
#Concatenamos en una sola ventana los videos
hor3=np.concatenate((gray,thresh1,thresh2,thresh3),axis=1)
hor4=np.concatenate((thresh4,thresh5,thresh6,thresh7),axis=1)
array2=np.concatenate((hor3,hor4),axis=0)
#se agrega un downsize debido a que la ventana de opencv  es mas grande que la resolucion de la pantalla
down_width=1920
down_height=1080
down_points=(down_width,down_height)
resize=cv.resize(array2,down_points,interpolation=cv.INTER_LINEAR)

#ventana emergente con pruebas de threshold
##    cv.imshow('pruebas thresholding',resize)
#PRUEBA C: TRANSFORMADORES DE HOUGH PARA LINEAS Y PARA CIRCULOS

##      alt 3 comentar y alt 4 para descomentar
##    #para esta parte se obtuvo el algoritmo de :https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/#:~:text=The%20Hough%20Transform%20is%20a,or%20distorted%20a%20little%20bit.
limim=frame
cirim=frame
lines=cv.HoughLines(edgesC,1,np.pi/180, 150)#,minLineLength=25,maxLineGap=10)


#realizar proteccion porque realmente no funciona como deberia y se traba tanto cuando se acelera mucho un cambio de valores X y Y sino que tambien cuando se tiene un acercamiento donde se pierdan los valores

for r_theta in lines:
    arr = list(np.array(r_theta[0], dtype=np.float64))
    r,theta = arr
    #se almacena el valor del coseno en a
    a = np.cos(theta)
    #se almacena el valor del seno en b
    b = np.sin(theta)
    #se almacena el valor de rcos en x0
    x0 = a*r
    #se almacena el valor de rsin en y0
    y0 = b*r
    #se almacena el valor redondeado de rcos-1000sin
    x1 = int(x0 + 1000*(-b))
    #se almacena el valor redondeado de rsin-1000cos
    y1= int(y0 + 1000*(a))
    #se almacena el valor redondeado de rcos +1000sin
    x2 = int(x0 - 1000*(-b))
    #se almacena el valor redondeado de rsin - 1000cos
    y2 = int(y0 -1000*(a))
    #se dibuja en la imagen la linea que va desde x1,y1 a x2,y2
    #(0,0,255) denota el color de la linea que se dibujara, ROJO
    lineas = cv.line(limim,(x1,y1),(x2,y2), (0,0,255),2)


#CIRCULOS
#obtener para la distancia minima entre centros mediante la imagen suavizada
rowsc = blur.shape[0]
#aplicar la transformada de HOUGH
circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, rowsc / 4,param1=200, param2=25,minRadius=40, maxRadius=100)
#blur: imagen entrada, HOUGH_GRADIENT: metodo de deteccion, dp: radio inverso de resolucion
#min_dist: rowsc/16 es la minima distancia entre los circulos detectados
#param1: umbral interno para el detector de bordes de canny
#param2: umbral para detecicon de centros
#minradius: 0 como default para ser detectado
#maxradius: 0 como default para ser detectado

#dibujamos los circulos detectados
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(cirim, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(cirim, center, radius, (255, 150, 255), 3)

#concatenacion de los videos en una sola ventana
#cv.imshow('pruebas transformadores de Hough',cirim)
os.chdir(direccion)
ret=True
if (ret == True):
    cv.imwrite('Theshold2_480.jpg', resize)
    if cv.waitKey(1) == ord('q'):
        exit()#break
else:
    exit()


if cv.waitKey(1)==ord('q'):
    exit()#break
#cap.release()
cv.destroyAllWindows()



