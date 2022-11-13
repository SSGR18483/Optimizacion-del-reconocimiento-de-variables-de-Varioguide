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


image_file='captura5off.jpg'
img = cv2.imread(image_file)

#plt.hist2d(img.flat,bins=100, range = (0,255))

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#regiones
#Aplicar multithreshold
thres=threshold_multiotsu(img)
regions = np.digitize(img,bins = thres)


#GRaficas

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Plotting the original image.
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(img.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thres:
    ax[1].axvline(thresh, color='r')

# Plotting the Multi Otsu result.
ax[2].imshow(regions, cmap='jet')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.subplots_adjust()

plt.show()


