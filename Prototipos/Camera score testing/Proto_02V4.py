#IMPORTE DE LIBRERIAS NECESARIAS
import keras
import matplotlib.pyplot
import numpy as np
import cv2 as cv
import glob
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image
from scipy import io

caso = 1
Entrenamiento = []
LabelsEntrenamiento = ['bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,bordes,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,lineas,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold,threshold']
Prueba = []
LabelsPrueba = ['bordes,bordes,bordes,bordes,bordes,lineas,lineas,lineas,lineas,lineas,threshold,threshold,threshold,threshold,threshold']
#Se llena de datos la primer lista
files_Train=glob.glob(r"C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Camera score testing/ProcessedTrain/IXR/*.jpg")
files_Val=glob.glob(r"C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Camera score testing/ProcessedVal/*.jpg")
for myFile in files_Train:
    print(myFile)
    image = Image.open(myFile).convert('RGB')
    image = np.array(image)
    if image is None or image.shape != (1080, 1920,3):
        print(f'This image is bad: {myFile} {image.shape if image is not None else "None"}')
    else:
        Entrenamiento.append(image)
for myFile2 in files_Val:
    print(myFile2)
    image2 = Image.open(myFile2).convert('RGB')
    image2 = np.array(image2)
    if image2 is None or image2.shape != (1080, 1920,3):
        print(f'This image is bad: {myFile2} {image2.shape if image2 is not None else "None"}')
    else:
        Prueba.append(image2)

#cv.imshow('imagen',Entrenamiento[21])
#if cv.waitKey(500)==ord('q'):
#        exit()
print('Entrenamiento con forma:', np.array(Entrenamiento).shape)
print('Prueba con forma:', np.array(Prueba).shape)
#cv.destroyAllWindows()
def model1():
    layer1=tf.keras.layers.Dense(5,activation='sigmoid',kernel_inizializer='he_uniform')
    model=tf.keras.models.Sequential()
    model.add(layer1)
    return model

