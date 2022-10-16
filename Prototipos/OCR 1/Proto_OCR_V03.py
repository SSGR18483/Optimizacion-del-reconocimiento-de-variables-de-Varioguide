#Prototipo EasyOCR vs Keras OCR vs Pytesseract
import easyocr
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract



# archivo = D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/
annot = pd.read_parquet('D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/annot.parquet')
imgs = pd.read_parquet('D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/img.parquet')
img_fns = glob('D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/Python/archive/train_val_images/train_images/*')

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

nI=22
#por pytesseract
print(pytesseract.image_to_string(img_fns[nI],lang='eng'))
fig,ax=plt.subplots(figsize=(10,10))
ax.imshow(plt.imread(img_fns[nI]))
ax.axis('off')
#plt.show()
#por easyocr
reader = easyocr.Reader(['en'], gpu = False)
results = reader.readtext(img_fns[nI])
print(results)
#pd.DataFrame(results, columns=['bbox','text','conf'])