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
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


def dibujo_contornos(hsv):
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    lower = np.array([39, 40, 38])
    upper = np.array([55, 57, 75])
    mask = cv2.inRange(blurred, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cnts= imutils.grab_contours(contours)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            cv2.drawContours(hsv, contour, -1, (0, 255, 0), 3)
            M = cv2.moments(contour)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(hsv,(cx,cy),7,(255,255,255),-1)
    return hsv,cx,cy
imagenf=dibujo_contornos(hsv)


#GRaficas

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))

# Plotting the original image.
ax[0].imshow(imagenf)
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# # multi-Otsu.
# ax[1].hist(img.ravel(), bins=255)
# ax[1].set_title('Histogram')
# for thresh in thres:
#     ax[1].axvline(thresh, color='r')

# Plotting the Multi Otsu result.
ax[1].imshow(img, cmap='jet')
ax[1].set_title('Mask')
ax[1].axis('off')

plt.subplots_adjust()
plt.show()



