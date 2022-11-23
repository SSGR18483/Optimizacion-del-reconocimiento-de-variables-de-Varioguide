# Universidad del Valle de Guatemala
# Santiago Galicia
# Programa para realizar pruebas de bloques individuales de programación
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
from skimage import measure
from imutils import contours
from skimage.filters import threshold_multiotsu
import imutils
import os
import tensorflow as tf
import glob

image_file='cutted.jpg'
img_color = cv2.imread(image_file)
img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

#SVHN TEST https://www.kaggle.com/code/yushg123/base-pipeline-for-mnist-98-8-accuracy/notebook  No sirvio porque necesita la info individual de los pixeles para funcionar
#MNIST video con keras https://www.youtube.com/watch?v=bte8Er0QhDg

def contorno_numeros(corte): # entra imagen normal y sale imagen normal con contornos de numeros.# procurar que sea la imagen cortada.
    roi = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi,(3,3),0)
    No_dig = 1
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.threshold(blur, 107, 510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w*h > 50) & (w*h < 600) :
            # cv2.rectangle(corte, (x-2, y-2), (x + w+2, y + h+2), color=(0, 255, 0), thickness=2)
            fig0 = roi[y-(w/2):y+(w/2), x-(h/2):x+(h/2)]
            digit = thresh[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            try:
                    filename = f"Digito{No_dig}.png"
                    status = cv2.imwrite(filename, fig0)
            except:
                print(f"Erorr! No. {No_dig}")
            finally:
                No_dig +=1
    return (corte)

figurita = contorno_numeros(img_color)