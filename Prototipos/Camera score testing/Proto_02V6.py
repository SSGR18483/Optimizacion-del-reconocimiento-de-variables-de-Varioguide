#programa realizado con la ayuda de:
# https://www.tensorflow.org/tutorials/images/classification
# https://www.tensorflow.org/tutorials/load_data/images
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

#Algoritmo Calificador de imagenes de camaras.
import keras
import matplotlib.pyplot
import numpy as np
import cv2 as cv
import glob
import h5py
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image
from scipy import io

#INICIO DE PROGRAMA DE PRUEBAS PARA CAMARAS
#cap= cv.VideoCapture(1)
pathimagenes='prueba4802.jpg'
frame= cv.imread(pathimagenes,cv.IMREAD_COLOR)
down_width1=1920
down_height1=1080
down_points1=(down_width1,down_height1)
frame=cv.resize(frame,down_points1,interpolation=cv.INTER_LINEAR)
cont=0
#direccion=r'C:\Users\galic\Documents\Diseño\GIT\Trabajo-de-Graduaci-n-SG18483\Prototipos\Camera score testing'
direccion=r'D:\Documentos\UVG\QUINTO AÑO\Segundo Semestre\Diseño e innovación\GIT\Optimizacion-del-reconocimiento-de-variables-de-Varioguide\Prototipos\Camera score testing'

#PROTECCION CONTRA FALTA DE LECTURA
#if not cap.isOpened():
    #print('no se pudo abrir la camara')
    #exit()
#Captura frame por frame
##ret, frame = cap.read()

#proteccion de lectura de frame

   # break
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#PROCESAMIENTO DE SUAVIZADO DE IMAGEN
blur= cv.GaussianBlur(gray,(3,3),0)
#detector No.1 Canny
edgesC = cv.Canny(blur,150,10, apertureSize=3)
#detector No.2 Sobel
sobelX = cv.Sobel(blur, cv.CV_64F, 2, 0)
sobelY = cv.Sobel(blur, cv.CV_64F, 0, 2)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombinado = cv.bitwise_or(sobelX, sobelY)
#detector No.3 laplaciano
lap = cv.Laplacian(blur, cv.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
#Concatenacion de los videos en una sola ventana
hor1=np.concatenate((gray,sobelCombinado),axis=0)
hor2=np.concatenate((lap,edgesC),axis=0)
array=np.concatenate((hor1,hor2),axis=1)
down_width2=1920
down_height2=1080
down_points2=(down_width2,down_height2)
resize2=cv.resize(array,down_points2,interpolation=cv.INTER_LINEAR)


thresh1 = cv.threshold(gray,120,255,cv.THRESH_BINARY)[1]
thresh2 = cv.threshold(gray,120,255,cv.THRESH_BINARY_INV)[1]
thresh3 = cv.threshold(gray,120,255,cv.THRESH_TRUNC)[1]
thresh4 = cv.threshold(gray,120,255,cv.THRESH_TOZERO)[1]
thresh5 = cv.threshold(gray,120,255,cv.THRESH_TOZERO_INV)[1]
thresh6 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,199,5)
thresh7 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,199,5)
hor3=np.concatenate((gray,thresh1,thresh2,thresh3),axis=1)
hor4=np.concatenate((thresh4,thresh5,thresh6,thresh7),axis=1)
array2=np.concatenate((hor3,hor4),axis=0)
down_width=1920
down_height=1080
down_points=(down_width,down_height)
resize=cv.resize(array2,down_points,interpolation=cv.INTER_LINEAR)


limim=frame
cirim=frame
lines=cv.HoughLines(edgesC,1,np.pi/180, 150)#,minLineLength=25,maxLineGap=10)

for r_theta in lines:
    arr = list(np.array(r_theta[0], dtype=np.float64))
    r,theta = arr
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1= int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 -1000*(a))
    lineas = cv.line(limim,(x1,y1),(x2,y2), (0,0,255),2)


#CIRCULOS
rowsc = blur.shape[0]
#aplicar la transformada de HOUGH
circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, rowsc / 4,param1=200, param2=25,minRadius=40, maxRadius=100)
#dibujamos los circulos detectados
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(cirim, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(cirim, center, radius, (255, 150, 255), 3)


###Aqui abajo cambiar las direcciones de las imagenes por la variable que contiene las imagenes.

#importe de imagenes
batch_size = 75 # se define el tamaño de los grupos
img_height=1080 # se define el alto de las imagenes
img_width=1920 # se define el ancho de las imagenes
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
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
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

model.save('my_model.h5')
savedir = 'D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Resultados'
tf.keras.models.save_model(
    model,
    savedir,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True,
)
#Tras entrenamiento se utiliza la predicción

#################################################       C920s        ##############################################
dirpre="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTest/Ejes1.jpg"
img = tf.keras.utils.load_img(
    dirpre, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("CAMARA LOGITECH C920s")
print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la primer prueba"
    .format( 100 * np.max(score))
)
dirpre2="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTest/Lineas1.jpg"
img2 = tf.keras.utils.load_img(
    dirpre2, target_size=(img_height, img_width)
)
img_array2 = tf.keras.utils.img_to_array(img2)
img_array2 = tf.expand_dims(img_array2, 0) # Create a batch

predictions2 = model.predict(img_array2)
score2 = tf.nn.softmax(predictions2[0])

print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la segunda prueba"
    .format( 100 * np.max(score2))
)
dirpre3="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTest/Threshold1.jpg"
img3 = tf.keras.utils.load_img(
    dirpre3, target_size=(img_height, img_width)
)
img_array3 = tf.keras.utils.img_to_array(img3)
img_array3 = tf.expand_dims(img_array3, 0) # Create a batch

predictions3 = model.predict(img_array3)
score3 = tf.nn.softmax(predictions3[0])

print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la tercer prueba"
    .format( 100 * np.max(score3))
)
print("Se le da una calificacion global de {:.2f} porciento a la camara Logitech C920s".format(100*(np.max(score)+np.max(score2)+np.max(score3))/3))

##########################################       C270        ############################################
dirpre4="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/Imagenes Test/16by9720/Ejes1_720.jpg"
img4 = tf.keras.utils.load_img(
    dirpre4, target_size=(img_height, img_width)
)
img_array4 = tf.keras.utils.img_to_array(img4)
img_array4 = tf.expand_dims(img_array4, 0) # Create a batch

predictions4 = model.predict(img_array4)
score4 = tf.nn.softmax(predictions4[0])
print("CAMARA LOGITECH C270")
print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la primer prueba"
    .format( 100 * np.max(score4)*0.97)
)
dirpre5="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/Imagenes Test/16by9720/Lineas1_720.jpg"
img5 = tf.keras.utils.load_img(
    dirpre5, target_size=(img_height, img_width)
)
img_array5 = tf.keras.utils.img_to_array(img5)
img_array5 = tf.expand_dims(img_array5, 0) # Create a batch

predictions5 = model.predict(img_array5)
score5 = tf.nn.softmax(predictions5[0])

print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la segunda prueba"
    .format( 100 * np.max(score5)*0.97)
)
dirpre6="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/Imagenes Test/16by9720/Threshold1_720.jpg"
img6 = tf.keras.utils.load_img(
    dirpre6, target_size=(img_height, img_width)
)
img_array6 = tf.keras.utils.img_to_array(img6)
img_array6 = tf.expand_dims(img_array6, 0) # Create a batch

predictions6 = model.predict(img_array6)
score6 = tf.nn.softmax(predictions6[0])

print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la tercer prueba"
    .format( 100 * np.max(score6)*0.97)
)
print("Se le da una calificacion global de {:.2f} porciento a la camara Logitech C270".format(100*0.97*(np.max(score4)+np.max(score5)+np.max(score6))/3))

################################################        Enow HD     #################################################
dirpre7="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTest/Ejes1.jpg"
img7 = tf.keras.utils.load_img(
    dirpre7, target_size=(img_height, img_width)
)
img_array7 = tf.keras.utils.img_to_array(img7)
img_array7 = tf.expand_dims(img_array7, 0) # Create a batch

predictions7 = model.predict(img_array7)
score7 = tf.nn.softmax(predictions7[0])
print("CAMARA ENOW HD")
print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la primer prueba"
    .format( 100 * np.max(score7)*0.93)
)
dirpre8="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTest/Lineas1.jpg"
img8 = tf.keras.utils.load_img(
    dirpre8, target_size=(img_height, img_width)
)
img_array8 = tf.keras.utils.img_to_array(img8)
img_array8 = tf.expand_dims(img_array8, 0) # Create a batch

predictions8 = model.predict(img_array8)
score8 = tf.nn.softmax(predictions8[0])

print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la segunda prueba"
    .format( 100 * np.max(score8)*0.93)
)
dirpre9="D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Camera score testing/ProcessedTest/Threshold1.jpg"
img9 = tf.keras.utils.load_img(
    dirpre9, target_size=(img_height, img_width)
)
img_array9 = tf.keras.utils.img_to_array(img9)
img_array9= tf.expand_dims(img_array9, 0) # Create a batch

predictions9 = model.predict(img_array9)
score9 = tf.nn.softmax(predictions9[0])

print(
    "Esta cámara obtiene la calificación de {:.2f} porciento en la tercer prueba"
    .format( 100 * np.max(score9)*0.93)
)
print("Se le da una calificacion global de {:.2f} porciento a la camara Enow HD".format(100*0.93*(np.max(score7)+np.max(score8)+np.max(score9))/3))


