#Universidad del Valle de Guatemala
#Facultad de Ingenieria
#Diseño e innovación
#Santiago Galicia - UVG18483
##Programa de prueba para determinar funcionalidad de procesamiento en vivo de video

#librerias
import numpy as np
import cv2 as cv


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
    cv.imshow('frame',gray)
    if cv.waitKey(1)==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
