#UVG Santiago Galicia 18483
#programa realizado con la ayuda de https://www.tensorflow.org/tutorials/images/classification
#IMPORTE DE LIBRERIAS NECESARIAS
import keras
import matplotlib.pyplot
import numpy as np
import cv2 as cv
import glob
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image
from scipy import io

#importe de imagenes
batch_size = 60
img_height=1080
img_width=1920
data_dir="C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Camera score testing/ProcessedTrain/IXR"
data_dir2="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTrain/IXR"


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir2,
  labels='inferred',
  label_mode='int',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir2,
  labels='inferred',
  label_mode='int',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# Se extraen las categorías como los valores únicos (diferentes) del array
# original de labels
class_names= train_ds.class_names
print(class_names)
#forma del espacio de imagenes
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#Modificacion de los datos
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#se necesitan normalizar los colores ya que 255 es demasiado para la red neuronal. se utilizara un rango de 0 a 1
num_classes = len(class_names)

model = tf.keras.models.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
#hyper parametros LR=0.01 y momento de 0.82 con 5 epochs
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.82,name='SGD',),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
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

#Tras entrenamiento se utiliza la predicción
dirpre="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTest/Lineas1.jpg"
#mg= tf.keras.utils.load_img()
img = tf.keras.utils.load_img(
    dirpre, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
