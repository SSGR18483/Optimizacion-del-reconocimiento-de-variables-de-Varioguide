#IMPORTE DE LIBRERIAS NECESARIAS
import keras
import numpy as np
import cv2 as cv
import glob
import tensorflow as tf
from matplotlib import pyplot
from matplotlib import rcParams
from scipy import io

caso = 1
Entrenamiento = []
LabelsEntrenamiento = []
Prueba = []
LabelsPrueba = []
for i in
def model1():
    layer1=tf.keras.layers.Dense(5,activation='sigmoid',kernel_inizializer='he_uniform')
    model=tf.keras.models.Sequential()
    model.add(layer1)


    opt= tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.84, name = 'SGS',)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
    history = model.fit(Entrenamiento, LabelsEntrenamiento, validation_data=(Prueba, LabelsPrueba), epochs=30)

    return history

_, train_acc = model.evaluate(LinesPTrain, LinesLTrain, verbose=0)
_, val_acc = model.evaluate(LinesPTest, LinesLTest, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, val_acc))
# Se grafica la evolución de la pérdida durante el entrenamiento y la
# validación
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# Se grafica la evolución de la exactitud durante el entrenamiento y la
# validación
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()