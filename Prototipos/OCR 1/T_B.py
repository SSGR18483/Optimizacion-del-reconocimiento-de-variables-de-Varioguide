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
import keras
import pandas as pd
import re
from cv2 import dnn_superres
from skimage import measure
from imutils import contours
from skimage.filters import threshold_multiotsu
import imutils
import os

image_file='cap1long.jpg'
img_color = cv2.imread(image_file)
img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

#SVHN TEST https://www.kaggle.com/code/yushg123/base-pipeline-for-mnist-98-8-accuracy/notebook

df_trian =
df_test =








