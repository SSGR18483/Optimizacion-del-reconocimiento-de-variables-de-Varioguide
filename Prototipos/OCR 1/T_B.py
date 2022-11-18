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

def dnnrescale(image,n):
    sr = dnn_superres.DnnSuperResImpl_create()
    imagednn =image
    path = "FSRCNN_x2.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn",n)
    img_dnresized = sr.upsample(imagednn)
    return img_dnresized



image_file='cap1long.jpg'
img_color = cv2.imread(image_file)
img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
DIGITSDICT = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}

def save_img(image,filename):
    status = cv2.imwrite(filename,image)
    return status
def contorno_numeros(corte):
    roi = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi,(3,3),0)
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.threshold(blur, 107, 510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w*h > 150) & (w*h < 600) :
            cv2.rectangle(corte, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            digit = thresh[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            # plt.imshow(padded_digit, cmap="gray")
            # plt.show()
            # xdigit = padded_digit.reshape(1, 784)
            # prediction = neigh.predict(xdigit)
            # print("prediction = ", prediction[0])
    return corte


#imagen de bgr a gris
roi_color = cv2.imread("cutted.jpg")
roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
# roi= dnnrescale(roi,2) #no lo detectaba adecuadamente.
RATIO = roi.shape[0] * 0.2
imagen_cortes = contorno_numeros(roi_color)
cv2.imshow("@", imagen_cortes)
cv2.waitKey(0)
estado= save_img(imagen_cortes,'pruebarapida.jpg')
# KERAS OCR # por el momento no me es util
pipeline = keras_ocr.pipeline.Pipeline()
image_keras=[keras_ocr.tools.read(imga) for imga in ['pruebarapida.jpg', 'cutted.jpg']]
imageneskeras=np.array(image_keras)
print(len(imageneskeras))
results = pipeline.recognize(imageneskeras)
print(pd.DataFrame(results[0], columns=['text', 'bbox']))
fig, axs = plt.subplots(nrows=len(imageneskeras), figsize = (20,20))
for ax, image,predictions in zip(axs,imageneskeras,results):
   keras_ocr.tools.drawAnnotations(image=image,
                                   predictions=predictions,
                                   ax=ax)
plt.show()
print(pd.DataFrame(results[1], columns=['text', 'bbox']))
# suavizando imagen
# roi = cv2.bilateralFilter(roi, 5, 30, 60)
roi =cv2.GaussianBlur(roi,(3,3),0)
# recortando
trimmed = roi[int(RATIO) :, int(RATIO) : roi.shape[1] - int(RATIO)]
roi_color = roi_color[int(RATIO) :, int(RATIO) : roi.shape[1] - int(RATIO)]
cv2.imshow("Blurred and Trimmed", trimmed)
cv2.waitKey(0)

edged = cv2.adaptiveThreshold(
    trimmed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5
)
cv2.imshow("Edged", edged)
cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1),)
eroded = cv2.erode(edged, kernel, iterations=1)

cv2.imshow("Eroded", eroded)
cv2.waitKey(0)

h = roi.shape[0]
ratio = int(h * 0.07)
eroded[-ratio:,] = 0
eroded[:, :ratio] = 0

cv2.imshow("Eroded + Black", eroded)
cv2.waitKey(0)

cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digits_cnts = []

canvas = trimmed.copy()
cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
cv2.imshow("All Contours", canvas)
cv2.waitKey(0)

canvas = trimmed.copy()
for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    if h> 20:
        digits_cnts += [cnt]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.drawContours(canvas, cnt, 0, (255, 255, 255), 1)

print(f"No. of Digit Contours: {len(digits_cnts)}")


cv2.imshow("xd", canvas)
cv2.waitKey(0)


sorted_digits = sorted  (digits_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])

canvas = trimmed.copy()


for i, cnt in enumerate(sorted_digits):
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
    cv2.putText(canvas, str(i), (x, y - 3), FONT, 0.3, (0, 0, 0), 1)

cv2.imshow("All Contours sorted", canvas)
cv2.waitKey(0)

digits = []
canvas = roi_color.copy()
for cnt in sorted_digits:
    (x, y, w, h) = cv2.boundingRect(cnt)
    roi = eroded[y : y + h, x : x + w]
    print(f"W:{w}, H:{h}")
    # convenience units dimensiones para reconocer 7 segmentos
    qW, qH = int(w * 0.25), int(h * 0.15)
    fractionH, halfH, fractionW = int(h * 0.05), int(h * 0.5), int(w * 0.25)

    # # seven segments in the order of wikipedia's illustration
    sevensegs = [
        ((0, 0), (w, qH)),  # a (top bar)
        ((w - qW, 0), (w, halfH)),  # b (upper right)
        ((w - qW, halfH), (w, h)),  # c (lower right)
        ((0, h - qH), (w, h)),  # d (lower bar)
        ((0, halfH), (qW, h)),  # e (lower left)
        ((0, 0), (qW, halfH)),  # f (upper left)
        #((0, halfH - fractionH), (w, halfH + fractionH)) # center
        (
            (0 + fractionW, halfH - fractionH),
            (w - fractionW, halfH + fractionH),
        ),  # center
    ]

    # initialize to off
    on = [1] * 2

    for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
        region = roi[p1y:p2y, p1x:p2x]
        print(
            f"{i}: Sum of 1: {np.sum(region == 255)}, Sum of 0: {np.sum(region == 0)}, Shape: {region.shape}, Size: {region.size}"
        )
        if np.sum(region == 255) > region.size * 0.5:
            on[i] = 1
        print(f"State of ON: {on}")

    digit = DIGITSDICT[tuple(on)]
    print(f"Digit is: {digit}")
    digits += [digit]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CYAN, 1)
    cv2.putText(canvas, str(digit), (x - 5, y + 6), FONT, 0.3, (0, 0, 0), 1)
    cv2.imshow("Digit", canvas)
    cv2.waitKey(0)

print(f"Digits on the token are: {digits}")