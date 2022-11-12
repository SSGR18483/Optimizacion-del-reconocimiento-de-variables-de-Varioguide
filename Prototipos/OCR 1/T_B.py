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


image_file='captura4off.jpg'
img = cv2.imread(image_file)
imgnp = np.array(Image.open(image_file))

while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue =
