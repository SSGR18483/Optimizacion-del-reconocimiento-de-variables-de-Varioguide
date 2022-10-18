#Prototipo EasyOCR vs Keras OCR vs Pytesseract
import easyocr
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import cv2 as cv
import os
import keras_ocr



# archivo = D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/
annot = pd.read_parquet('D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/annot.parquet')
imgs = pd.read_parquet('D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/img.parquet')
img_fns = glob('D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/train_val_images/train_images/*.jpg')
capturadir= r'D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/OCR 1/captura1.jpg'
captura=cv.imread(capturadir,0)
# Ploteamos ejemplo

# fig, ax = plt.subplots(figsize = (10,10))
# ax.imshow(plt.imread(img_fns[0]))
# ax.axis('off')
# plt.show()
# image_id = img_fns[0].split('/')[-1].split('.')[0]
# annot.query('image_id == @image_id')

#despliegue para las primeras 25 imagenes
# fig, axs = plt.subplots(5, 5, figsize=(20, 20))
# axs = axs.flatten()
# for i in range(25):
#     axs[i].imshow(plt.imread(img_fns[i]))
#     axs[i].axis('off')
#     image_id = img_fns[i].split('/')[-1].rstrip('.jpg')
#     n_annot = len(annot.query('image_id == @image_id'))
#     axs[i].set_title(f'{image_id} - {n_annot}')
# plt.show()

nI=6
#por pytesseract
lectura=(pytesseract.image_to_string(captura,lang='eng'))
print(lectura)
fig,ax=plt.subplots(figsize=(10,10))
ax.imshow(plt.imread(captura))
ax.axis('off')
plt.show()
#por easyocr
reader = easyocr.Reader(['en'])
img_easy = img_fns[nI] #aqui se coloca la imagen
results = reader.readtext(captura)
pd.DataFrame(results, columns=['bbox','text','conf'])
#por keras OCR
pipeline = keras_ocr.pipeline.Pipeline()
results = pipeline.recognize(captura)
pd.DataFrame(results[0], columns=['text', 'bbox'])
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(plt.captura, results[0], ax=ax)
ax.set_title('Keras OCR Result Example')
# plt.show()