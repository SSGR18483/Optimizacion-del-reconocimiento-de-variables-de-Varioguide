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

caso = 1
nb_train_samples = 60
nb_validation_samples = 15
epochs = 10
batch_size = 16
#pre definir las listas que se van a utilizar
Entrenamiento = []
LabelsEntrenamiento = ['bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','bordes','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','lineas','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold','threshold']
LabelsEntrenamiento = np.array(LabelsEntrenamiento)
Prueba = []
LabelsPrueba = ['bordes','bordes','bordes','bordes','bordes','lineas','lineas','lineas','lineas','lineas','threshold','threshold','threshold','threshold','threshold']
LabelsPrueba = np.array(LabelsPrueba)

#direcciones de los datos
#PC
files_Train=glob.glob(r"D:\Documentos\UVG\QUINTO AÑO\Segundo Semestre\Diseño e innovación\GIT\Optimizacion-del-reconocimiento-de-variables-de-Varioguide\Prototipos\Camera score testing\ProcessedTrain\IXR/*.jpg")
files_Val=glob.glob(r"D:\Documentos\UVG\QUINTO AÑO\Segundo Semestre\Diseño e innovación\GIT\Optimizacion-del-reconocimiento-de-variables-de-Varioguide\Prototipos\Camera score testing\ProcessedVal/*.jpg")
#Laptop
#files_Train=glob.glob(r"C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Camera score testing/ProcessedTrain/IXR/*.jpg")
#files_Val=glob.glob(r"C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Camera score testing/ProcessedVal/*.jpg")

#importe de imagenes
batch_size = 60
img_height=1080
img_width=1920
#primer forma de cargar las imagenes
for myFile in files_Train:
    print(myFile)
    image = Image.open(myFile).convert('RGB')
    image = np.array(image)
    if image is None or image.shape != (1080, 1920,3):
        print(f'la imagen no cumple los requisitos de resolución: {myFile} {image.shape if image is not None else "None"}')
    else:
        Entrenamiento.append(image)
for myFile2 in files_Val:
    print(myFile2)
    image2 = Image.open(myFile2).convert('RGB')
    image2 = np.array(image2)
    if image2 is None or image2.shape != (1080, 1920,3):
        print(f'la imagen no cumple los requisitos de resolución: {myFile2} {image2.shape if image2 is not None else "None"}')
    else:
        Prueba.append(image2)

print('Entrenamiento con forma:', np.array(Entrenamiento).shape)
print('Prueba con forma:', np.array(Prueba).shape)
data_dir="C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Camera score testing/ProcessedTrain/IXR"
data_dir2="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTrain/IXR"

#Segunda forma de recoger las imagenes
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
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.3,name='SGD',),
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
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Tras entrenamiento se utiliza la predicción
#dirpre=
#mg= tf.keras.utils.load_img()
# classes = np.unique(LabelsEntrenamiento)
# # Se crean arrays de ceros con las mismas dimensiones de los arrays originales
# # de labels
# YTrainLabel = np.zeros_like(LabelsEntrenamiento)
# YTestLabel = np.zeros_like(LabelsPrueba)
#
# # Se convierte la categoría desde una letra 'A', 'B', 'C' a un número 0, 1 o 2
# # respectivamente
# for nc in range(len(classes)):
#     YTrainLabel[LabelsEntrenamiento == classes[nc]] = nc
#     YTestLabel[LabelsPrueba == classes[nc]] = nc
#
# # Se elimina la dimensión "adicional" de los vectores para poder hacer un
# # one-hot encoding con la misma en Keras
# #YTrainLabel = YTrainLabel.reshape(-1)
# #YTestLabel = YTestLabel.reshape(-1)
#
# # Se efectúa un one-hot encoding para las labels
# LTrain = tf.keras.utils.to_categorical(YTrainLabel)
# LTest = tf.keras.utils.to_categorical(YTestLabel)
#
# #Definicion
# layer1=tf.keras.layers.Dense(50,activation='sigmoid')#,input_shape= (1080,1920,3))
# layer2=tf.keras.layers.Dense(3,activation='relu')
# model=tf.keras.models.Sequential()
# model.add(layer1)
#
#
# #Entrenamiento
# opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.84,name='SGD',)
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=opt ,metrics=['accuracy'])
# history = model.fit(Entrenamiento,(LTrain),validation_data=(Prueba,(LTest)),epochs=30)
#
# #Evaluación del modelo
# _, train_acc = model.evaluate(Entrenamiento, LTrain, verbose = 0)
# _, val_acc = model.evaluate(Prueba, LTest, verbose = 0)
# print('Train: %.3f, Test: %.3f' % (train_acc, val_acc))
# # Se grafica la evolución de la pérdida durante el entrenamiento y la
# # validación
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # Se grafica la evolución de la exactitud durante el entrenamiento y la
# # validación
# plt.subplot(212)
# plt.title('Accuracy')
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show()
#
#
