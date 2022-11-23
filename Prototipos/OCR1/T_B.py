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

# image_file='cap1long.jpg'
# img_color = cv2.imread(image_file)
# img=cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

#SVHN TEST https://www.kaggle.com/code/yushg123/base-pipeline-for-mnist-98-8-accuracy/notebook  No sirvio porque necesita la info individual de los pixeles para funcionar
#MNIST video con keras https://www.youtube.com/watch?v=bte8Er0QhDg


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

xtrain = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(1000, activation ='tanh'))
model.add(tf.keras.layers.Dense(500, activation ='relu'))
model.add(tf.keras.layers.Dense(100, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(10,activation = 'softmax'))
epochs = 5
model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics= ['accuracy'])
history=model.fit(x_train, y_train, epochs=epochs,validation_data=(x_test,y_test))
# model.save('mnist.model')
# model = tf.keras.models.load_model('mnist.model')

# loss, accuracy = model.evaluate(x_test, y_test)

def graf_DNN(history,epochs): #funcion que recibe un model fit con epochs y grafica el desempeño del modelo
    #recomendable utilizar un model.fit con datos de entrenamiento, epochs y los datos de validacion
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy de entrenamiento vs validacion')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Perdida de entrenamiento y validacion')
    plt.show()
    return


def no_proces(imagen): #funcion que recibe imagen y que la regresa en blanco y negro con dimensiones de 28x28
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    (thresh, imagen) = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    imagen = cv2.resize(imagen, (28, 28), interpolation=cv2.INTER_LINEAR)
    imagen = np.array([imagen])
    return imagen

image_no=1
while os.path.isfile(f"digit{image_no}.png"):
    try:
        path = f"digit{image_no}.png"
        img = cv2.imread(path)
        img = no_proces(img)
        # plt.imshow(img[0],cmap=plt.cm.binary)
        # plt.show()
        prediction = model.predict(img)
        print(f"el numero es probablemente un {np.argmax(prediction)}")
        if image_no ==1:
            digit1 = np.argmax(prediction)
        elif image_no ==2:
            digit2 =np.argmax(prediction)
        else:
            print("Error")

    except:
        print("Error!")
    finally:
        image_no +=1
digitos= digit1+(digit2/10)
print(float(digitos))

