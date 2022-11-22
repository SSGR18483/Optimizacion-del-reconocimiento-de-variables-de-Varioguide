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

image_file='cap1long.jpg'
img_color = cv2.imread(image_file)
img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

#SVHN TEST https://www.kaggle.com/code/yushg123/base-pipeline-for-mnist-98-8-accuracy/notebook  No sirvio porque necesita la info individual de los pixeles para funcionar

#MNIST video con keras https://www.youtube.com/watch?v=bte8Er0QhDg

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

xtrain = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(528, activation ='relu'))
model.add(tf.keras.layers.Dense(150, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(10,activation = 'softmax'))
model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics= ['accuracy'])
model.fit(x_train, y_train)
# model.save('mnist.model')
# model = tf.keras.models.load_model('mnist.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

image_no=0
while os.path.isfile(f"TB/digit{image_no}.png"):
    try:
        img = cv2.imread(f"TB/digit{image_no}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"el numero es probablemente un {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_no +=1




