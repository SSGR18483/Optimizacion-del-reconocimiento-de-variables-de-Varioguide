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


image_file='cap1long.jpg'
img = cv2.imread(image_file)
pic=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def mask_manual(img,modo):
    pic=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if modo == 1:
        mask1 = cv2.inRange(pic, (41, 41, 35), (49, 51, 76))
        mask2 = cv2.inRange(pic, (36, 36, 36), (55, 58, 76))
        mask = cv2.bitwise_or(mask1, mask2)
    elif modo == 2:
        mask = cv2.inRange(pic,(39,40,38),(55,57,75))
    elif modo == 3:
        mask = cv2.inRange(pic,(36,38,38),(60,60,70))
    return mask
maskar=mask_manual(img,3)
show= plt.imshow(maskar)
plt.title('Example',                                    fontweight ="bold")
plt.show()