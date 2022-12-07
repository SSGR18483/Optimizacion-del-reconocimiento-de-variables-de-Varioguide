#
#     __  ___    ________
#    / / / / |  / / ____/
#   / / / /| | / / / __
#  / /_/ / | |/ / /_/ /
#  \____/  |___/\____/
#
#  GUI
#
# UVG
# Prototipo de GUI para el diseño final
# Santiago Galicia
# Carnet 18483

## Importe de librerias
import numpy as np
import cv2
import pytesseract
from matplotlib import pyplot as plt
from PIL import ImageTk
from PIL import Image
import PIL as PIL
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import keras_ocr
from cv2 import dnn_superres
import tensorflow as tf
import os
import re
import socket


###################################################################
####################### Definicion variables globales #######################
###################################################################
global ret
global connection
global client_address
global digitos
digitos =0

###################################################################
####################### Definicion general #######################
###################################################################

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
HOST = "192.168.130.176"  # Standard loopback interface address (localhost)
PORT = 80  # Port to listen on (non-privileged ports are > 1023)

server_address = (HOST, PORT)
sock.bind(server_address)
juntas = [[1.1, digitos, 1.1]]
if __name__ == "__main__":
    # Listen for incoming connections
    sock.listen(1)
    print("Esperando conexión...")
    connection, client_address = sock.accept()
    print("defined")
    num_juntas = len(juntas)
###################################################################
############################# Funciones OCR #######################
###################################################################
def obtenercaptura():
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    vidcap.set(cv2.CAP_PROP_FOCUS, 80)
    widthc = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    heightc = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(widthc, heightc)
    # check if connection with camera is successfully
    if vidcap.isOpened():
        ret, frame = vidcap.read()  # capture a frame from live video

        # check whether frame is successfully captured
        if ret:
            print("se pudo capturar el frame")
        # print error if frame capturing was unsuccessful
        else:
            print("Error : no se capturo el frame")

    # print error if the connection with camera is unsuccessful
    else:
        print("Cannot open camera")
    return frame


# MANEJO DE DATOS DE LOS SISTEMAS
def Handdle(String):
    if String.find('Joint 1') >= 0 or String.find('int 1') >= 0:
        Joint = 1
        aftermath = 'Se leyó adecuadamente'
    elif String.find('Joint 2') >= 0 or String.find('int 2') >= 0:
        Joint = 2
        aftermath = 'Se leyó adecuadamente'
    elif String.find('Joint 3') >= 0 or String.find('int 3') >= 0:
        Joint = 3
        aftermath = 'Se leyó adecuadamente'
    else:
        Joint = 0
        aftermath = 'No se pudo leer adecuadamente, intente de nuevo'
    return Joint, aftermath


def signohanddle(string):
    if string.find(' - ') >= 0 or string.find('-') >= 0:
        signo = -1
    else:
        signo = 1
    return signo


# Procesamiento de imágenes
# inversión
def inversion(img):
    inverted_image = cv2.bitwise_not(img)
    return inverted_image


# Redimensionamiento
def dnnrescale(image, n):
    sr = dnn_superres.DnnSuperResImpl_create()
    imagednn = image
    path = "FSRCNN_x2.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn", n)
    img_dnresized = sr.upsample(imagednn)
    return img_dnresized


# Reescalada
def rescale(image, width, height):
    down_points1 = (width, height)
    img_resized = cv2.resize(image, down_points1, interpolation=cv2.INTER_LINEAR)
    return img_resized


# Escala de grises
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Thresholding
def umbral(image):
    thresh = cv2.threshold(image, 107, 510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


# Noise removal
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


# adelgazando la fuente
def thin(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=2)
    image = cv2.bitwise_not(image)
    return image


# haciendo mas grande la fuente
def thick(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=2)
    image = cv2.bitwise_not(image)
    return image


# Rotation/deskewing
def getSkewAngle(image) -> float:
    newImage = image.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    largestContour = contours[0]
    # print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return (-1.0 * angle, newImage)


def rotation(image, angle: float):
    newImage = image.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(image):
    angle = getSkewAngle(image)[0]
    return rotation(image, -1.0 * angle)


# quitar bordes
def quitar_bordes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 6, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y + h, x:x + w]
    return crop


# Agregar bordes que hacen falta
def agregar_borde(image):
    color = [0, 0, 0]
    top, bottom, left, right = [50] * 4
    image_extbord = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_extbord


def save_img(image, filename):
    status = cv2.imwrite(filename, image)
    return status


def crop_img(image, cx, cy):  # ,x,y):#imgnp
    fig0 = image[cy - 40:cy + 40, cx - 60:cx + 90, :]
    return fig0


def Blurred(image):
    image = cv2.GaussianBlur(image, (5, 5), 2)
    return image


def mask_manual(entry, modo):  # https://www.youtube.com/watch?v=YRb48EUk6Dk ///
    # https://notebook.community/ricklon/opencvraspberrypi/notebook/openCV%20color%20detection
    pic = cv2.cvtColor(entry, cv2.COLOR_BGR2RGB)
    if modo == 1:
        mask1 = cv2.inRange(pic, (41, 41, 35), (49, 51, 76))
        mask2 = cv2.inRange(pic, (36, 36, 36), (55, 58, 76))
        mask = cv2.bitwise_or(mask1, mask2)
    elif modo == 2:
        mask = cv2.inRange(pic, (39, 40, 38), (55, 57, 75))
    elif modo == 3:
        mask = cv2.inRange(pic, (38, 38, 36), (73, 73, 72))
    else:
        mask = cv2.inRange(pic, (0, 0, 0), (255, 255, 255))
    return mask


def dibujo_contornos(picture):
    imagen = picture
    blurred = cv2.GaussianBlur(imagen, (5, 5), 0)
    #######################################################
    # lower = np.array([39, 40, 38])
    # lower = np.array([38,38,36])# en caso sea ultimas fotos
    # lower = np.array([45, 50, 49])  # en caso sea live
    # lower = np.array([12, 13, 12])  # en caso sea live apagado
    lower = np.array([37, 35, 37])  # en caso sea live
    # lower = np.array([9, 5, 4])
    # upper = np.array([55, 57, 75])
    # upper = np.array([73,73,72]) #en caso sea ultimas fotos
    upper = np.array([56, 63, 72])  # en caso sea live
    # upper = np.array([17, 17, 28])
    # upper = np.array([57, 65, 72])  # en caso sea live
    # upper = np.array([25, 24, 46])  # en caso sea live apagado
    #######################################################
    mask = cv2.inRange(blurred, lower, upper)
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                # cv2.drawContours(guardar, contour, -1, (0, 255, 0), 3)
                # estado = cv2.imwrite('Primercontorno.jpg', cv2.cvtColor(guardar,cv2.COLOR_RGB2BGR))
                M = cv2.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # cv2.circle(imagen,(cx,cy),7,(255,255,255),-1)
    except:
        cx = 0
        cy = 0
    return imagen, cx, cy


def recorte_inicial(image, cx, cy):
    # fig0=image[ cy-505:cy+505,cx-665:cx+640,:]
    fig0 = image[cy - 540:cy + 540, cx - 680:cx + 640, :]
    # fig0 = image[cy - 485:cy + 485, cx - 660:cx + 631, :]
    # fig0 = image[cy - 456:cy + 456, cx - 577:cx + 577, :]
    return fig0


def contorno_numeros(
        corte):  # entra imagen normal y sale imagen normal con contornos de numeros.# procurar que sea la imagen cortada.
    roi = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    No_dig = 1
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.threshold(blur, 107, 510, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel = np.ones((3, 4), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            try:
                if (w * h > 200) & (w * h < 600):
                    # cv2.rectangle(corteguardar, (x-(2), y-(2)), (x + w+(2), y + h+(2)), color=(0, 255, 0), thickness=2)
                    # estado = cv2.imwrite('contornosnum.jpg', corteguardar)
                    fig0 = roi[y - 2:y + h + 2, x - 3:x + w + 3]
                    fig0 = cv2.cvtColor(fig0, cv2.COLOR_GRAY2RGB)
                    filename = f"digit{No_dig}.png"
                    status = cv2.imwrite(filename, fig0)
                    No_dig += 1
            except:
                print(f"Error al procesar No. {No_dig}\n")
    except:
        print(f"Error al procesar contornos de número")
        pass
    return corte


def keras_dr(
        ad1):  # funcion que recibe la direccion de la imagen guardada o una variable y entrega un string con los digitos analizados de keras.
    a = 0
    try:
        pipeline = keras_ocr.pipeline.Pipeline()
        image = [keras_ocr.tools.read(ad1)]
        imagenkeras = np.array(image)
        resultado = pipeline.recognize(imagenkeras)
        # a = pd.DataFrame(resultado[0], columns=['text','bbox'])
    finally:
        a = resultado[0][0]
        sfj = a[0]
    return sfj


# ejemplo keras:
# img_ker='cutted.jpg'
# sfj= keras_dr('cutted.jpg')
# print(sfj)
# print(sfj[0:2]) #[0] es  para el primer digito. [1] es para el decimal.

def no_proces(imagen):  # funcion que recibe imagen y que la regresa en blanco y negro con dimensiones de 28x28
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    (thresh, imagen) = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    imagen = cv2.resize(imagen, (28, 28), interpolation=cv2.INTER_LINEAR)
    imagen = np.array([imagen])
    return imagen


def graf_DNN(history, epochs):  # función que recibe un model fit con epochs y grafica el desempeño del modelo
    # recomendable utilizar un model.fit con datos de entrenamiento, epochs y los datos de validación
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
    plt.title('Perdida de entrenamiento y validación')
    plt.show()
    return


def processsing(img):
    CUT_GR = grayscale(img)
    CUT_BLR = Blurred(CUT_GR)
    CUT_UM = umbral(CUT_BLR)
    CUT_INV = inversion(CUT_UM)
    CUT_THIN = thin(CUT_INV)
    return CUT_BLR

###################################################################
############################# Funciones TCP #######################
###################################################################

def TCPsend(dic):

    j1 = dic[0]
    s = dic[1]
    No = dic[2]
    # mensaje = {"data":[j1,s,No]}#J1,S,NO:
    mensaje = "{data: [" + f"{j1}" + "," + f"{s}" + "," + f"{No}" + "]}"
    mensaje = mensaje.encode("ascii")  # se configura en ascci
    connection.sendall(mensaje)  # manda el mensaje
    return mensaje


def TCPreceive():
    # Receive the data in small chunks and retransmit it
    recv_data = connection.recv(16)
    return recv_data
###################################################################
############################# Funciones GUI #######################
###################################################################
root = Tk()
root.geometry("1000x750")

# img=PIL.Image.open("D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
img = PIL.Image.open(
    "C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
bg = ImageTk.PhotoImage(img)

image_file = 'J2[8.2].jpg'


def exitclick():
    root.destroy()


# limpiar
def cleart():
    mytext1.delete(1.0, END)


def mostrarJ():
    # CASO
    CASO = 1
    if CASO == 1:
        img = obtenercaptura()
    elif CASO == 2:
        img = cv2.imread(image_file)
    else:
        print("no se pudo xd")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img,cmap=plt.cm.binary)
    plt.show()
    # estado = save_img(img,'normal.jpg')
    status = cv2.imwrite('INICIAL1.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # img es la imagen original
    Corte1 = recorte_inicial(img, 960, 540)
    # status = cv2.imwrite('corteini.jpg',cv2.cvtColor(Corte1, cv2.COLOR_RGB2BGR) #pa guardar con color

    # Reconocimiento de Caracteres por medio de Tesseract
    res_img = dnnrescale(Corte1, 2)  # width=3840,height=2160)#2x 1080x1920
    gray_img = grayscale(res_img)
    blur_img = Blurred(gray_img)
    umb_img = umbral(blur_img)
    nonoise_img = noise_removal(umb_img)  # imagen sin ruido
    thin_img = thin(nonoise_img)  # imagen con letras mas delgadas
    thick_img = thick(nonoise_img)  # imagen con letras mas gruesas
    ocr_result = pytesseract.image_to_string(nonoise_img)
    estado = save_img(umb_img, 'processed.jpg')

    # Lectura de Junta
    Joints, mensajito = Handdle(str(ocr_result))
    if mensajito == 'No se pudo leer adecuadamente, intente de nuevo':
        try:
            ocr_result = pytesseract.image_to_string(gray_img)
            print(ocr_result)
        except:
            print("no se pudo arreglar dvd")
    else:
        pass
    Joints, mensajito = Handdle(str(ocr_result))

    ###################################################################################################################
    ###############################Determinacion de digitos angulo y desplazamiento############################################
    ###################################################################################################################

    if Joints == 2:
        try:
            trim_angle = Corte1[540 - 310:540 - 90, 660 + 240:660 + 540, :]
            trim_disp = Corte1[540 - 90:540 + 130, 660 + 240:660 + 540, :]
            image_angle_j2, cx2, cy2 = dibujo_contornos(trim_angle)
            image_angle = crop_img(image_angle_j2, cx2, cy2)
            image_displacement_j2, cx3, cy3 = dibujo_contornos(trim_disp)
            image_displacement = crop_img(image_displacement_j2, cx3, cy3)
            estado = save_img(image_angle, 'cutted.jpg')
            estado = save_img(image_displacement, 'despcutted.jpg')
        except:
            print("no se pudo obtener el corte de la junta 2")
            pass
    elif Joints == 0 or Joints == 1 or Joints == 3:
        try:
            image_angles, cx, cy = dibujo_contornos(Corte1)
            trim = crop_img(image_angles, cx, cy)
            estado = save_img(trim, 'cutted.jpg')
        except:
            print("no se pudo obtener el corte de la junta 013")
            pass
    ###################################################################################################################
    ##############################################Lectura de digitos###################################################
    ###################################################################################################################
    image_archivo = 'cutted.jpg'
    img_color_cut = cv2.imread(image_archivo)
    figurita = contorno_numeros(img_color_cut)
    print("-----------------")
    print("Lectura:")
    print(mensajito)
    print("\n")
    print("Junta:")
    print(Joints)
    print("\n")
    print("-----------------")

    if Joints == 2:
        try:
            file_disp = 'despcutted.jpg'
            img_disp = cv2.imread(file_disp)
            # contornosdesp = contorno_numeros(img_disp)
            procesodesp = processsing(img_disp)
            ocr_disp = pytesseract.image_to_string(procesodesp)
        except:
            ocr_disp = "0.00"

    processed_sign = processsing(img_color_cut)
    ocr_result_sign2 = pytesseract.image_to_string(
        processed_sign)
    ocr_result_sign = pytesseract.image_to_string(
        processed_sign,config='digits')  # ,config='--psm 12 --oem 3 -c tessedit_char_whitelist=0123456789.-')
    signo = signohanddle(str(ocr_result_sign2))
    print("-----------------")
    print(signo)
    ###################################################################################################################
    ##############################################Procesamiento de string##############################################
    ###################################################################################################################
    if Joints == 1 or Joints == 3 or Joints == 0:
        try:
            texto = ocr_result_sign
            # OCR_test_adjustment = float(re.sub('\D', texto))
            print(texto)
            OCR_adjustment = texto.replace(('°', '').replace('º', '').replace(' ', '').replace('$', '5').replace('\n', '').replace('PO', ''))
        except ValueError:

            texto = ocr_result_sign
            print(texto)
            # OCR_test_adjustment = float(re.sub(r'[^0-9]', '', texto))
            OCR_adjustment = 0.0
        except:
            texto = ocr_result_sign
            print(texto)
            OCR_adjustment = 0.0#float(re.sub(r'[^0-9]', '', texto))
                # texto.replace('°', '').replace('º', '').replace(' ', '').replace('$', '5').replace('\n', '').replace(
                #     'PO', ''))

    elif Joints == 2:
        try:
            OCR_adjustment = float(ocr_result_sign[0:4].replace('º\n', '').replace('º', '').replace(' ', ''))
            OCR_displacement = float(
                ocr_disp[0:4].replace('mm', '').replace('nm', '').replace('mn', '').replace(' ', ''))
            OCR_test_adjustment = float(re.sub('\D', ocr_result_sign))
            OCR_test_displacement = float(re.sub('\D', ocr_disp))
            print(f"displacement: {OCR_test_displacement}")
        except:
            OCR_adjustment = float(ocr_result_sign[0:3].replace('º\n', '').replace('º', '').replace(' ', ''))
            OCR_displacement = float(
                ocr_disp[0:3].replace('mm', '').replace('nm', '').replace('mn', '').replace(' ', ''))
            OCR_test_adjustment = float(re.sub('\D', ocr_result_sign))

            OCR_test_displacement = float(re.sub('\D', ocr_disp))
            print(f"displacement: {OCR_test_displacement}")
    else:
        print("Error junta reconocida errónea")
    # print(OCR_test_adjustment)
    ###################################################################################################################
    ####################################################MODELO MNIST###################################################
    ###################################################################################################################

    mnist = tf.keras.datasets.mnist
    # importe de set de datos y proceso de datos
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train / 255
    X_test = x_test / 255
    X_train = X_train.reshape(-1, 28, 28, 1)  # training set
    X_test = X_test.reshape(-1, 28, 28, 1)
    # model= tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(120, activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    # epochs=20
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # history=model.fit(X_train,y_train, epochs=epochs,validation_data=(X_test,y_test))
    # model.save('cnn.model_L')
    # model.save('cnn.model')

    model = tf.keras.models.load_model('cnn.model_L')
    # model = tf.keras.models.load_model('cnn.model')
    loss, accuracy = model.evaluate(X_test, y_test)
    # graf_DNN(history,epochs)
    digit0 = 0;
    digit1 = 0;
    digit2 = 0;
    digit3 = 0;
    # model = tf.keras.models.load_model('cnn.model')
    image_no = 1
    while os.path.isfile(f"digit{image_no}.png"):
        try:
            path = f"digit{image_no}.png"
            img = cv2.imread(path)
            img = no_proces(img)
            # plt.imshow(img[0],cmap=plt.cm.binary)
            # plt.show()
            prediction = model.predict(img)
            # print(f"el numero es probablemente un {np.argmax(prediction)}")
            if image_no == 1:
                digit2 = np.argmax(prediction)
            elif image_no == 2:
                digit1 = np.argmax(prediction)
            elif image_no == 3:
                digit0 = np.argmax(prediction)
            elif image_no == 4:
                digit3 = np.argmax(prediction)
            else:
                print("Error, no se almacenaron bien las imágenes")
        except:
            print("Error al reconocer MNIST")
        finally:
            image_no += 1
    digitos = float(digit3*100+digit0 * 10 + digit1 + (digit2 / 10))
    print(digitos)
    # HACER GLOBAL LA VARIABLE QUE AL FINAL CONTENGA LOS DÍGITOS
    if Joints == 1 or Joints == 3 or Joints == 0:
        if digitos == OCR_adjustment:
            print("````````````````````````````````````````````````````````````````````````")
            print(f"Se pudo leer adecuadamente el ángulo de arreglo de la junta 1 y es: {signo * digitos}")
            print("........................................................................")
            oracion = f"Se pudo leer adecuadamente el ángulo de arreglo de la junta 1 y es: {signo * digitos}"
        elif digitos != OCR_adjustment:
            print("````````````````````````````````````````````````````````````````````````")
            # print(f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {signo*digitos}")
            print(f"Se pudo leer adecuadamente el ángulo de arreglo de la junta 1 y es: {signo * digitos }")
            print("........................................................................")
            oracion = f"Se pudo leer adecuadamente el ángulo de arreglo de la junta 1 y es: {signo * digitos}"
            # oracion = f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {signo*OCR_adjustment}"
        else:
            print("no funciono bien ")
    elif Joints == 2:
        if digitos == OCR_adjustment:
            print("````````````````````````````````````````````````````````````````````````")
            print(
                f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {signo * digitos}º con desplazamiento de: {OCR_displacement}mm")
            print("........................................................................")
            oracion = f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {signo * digitos}º con desplazamiento de: {OCR_displacement}mm"
        elif digitos != OCR_adjustment:
            print("````````````````````````````````````````````````````````````````````````")
            print(
                f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {OCR_adjustment}º con desplazamiento de: {OCR_displacement}mm")
            print("........................................................................")
            oracion = f"{mensajito} el ángulo de arreglo de la junta {Joints} y es: {OCR_adjustment}º con desplazamiento de: {OCR_displacement}mm"
        else:
            print("no funciono bien ")
    mytext1.insert(0.1, str(oracion))  # 0.1 porque es donde empieza a poner el texto



def showimagencort():
    if os.path.isfile(f"cutted.jpg"):
        # cortada = PIL.Image.open("D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/OCR1/cutted.jpg")
        cortada = PIL.Image.open(
            "C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/OCR1/cutted.jpg")
        trimmed = ImageTk.PhotoImage(cortada)
        imagebox = Text(root, width=25, height=5)
        imagebox.place(x=0, y=660)
        imagebox.window_create(tk.END, window=tk.Label(imagebox, image=trimmed))
    else:
        print("No se encontró la imagen de elipse con dígitos.")
    return
juntas = [[1.1, 0.2, 1.1]]

def TCP_button():
    for index in range(0, num_juntas):
        print("Mandar mensaje...")
        msg = TCPsend(juntas[index])
        print("Recibir mensaje...")
        msg_recv = TCPreceive()

        print(msg)
        print("\n")
        print(msg_recv)
        print("\n")
    return

# Create Canvas
canvas1 = Canvas(root, width=1000, height=750)
canvas1.pack(fill="both", expand=True)
# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")
# Add Text
canvas1.create_text(500, 25, text="Sistema OCR HUMANA ", font=("Courier", 30, 'bold'), fill="black")

# crear widget de texto
mytext1 = Text(root, width=40, height=5, font=("Tahoma", 12, 'bold'))
mytext1.place(x=300, y=345)

# Create Buttons
button1 = Button(root, text="Reconocer Valores", command=mostrarJ)
button1.place(x=25, y=50, width=130, height=50)
button2 = Button(root, text="Salir", command=exitclick)
button2.place(x=900, y=700, width=100, height=50)
button3 = Button(root, text="Enviar Valores",command= TCP_button)
button3.place(x=850, y=50, width=100, height=50)
button4 = Button(root, text="Limpiar", command=cleart)
button4.place(x=705, y=395, width=50, height=50)
button5 = Button(root, text="Mostar digito", command=showimagencort)
button5.place(x=25, y=250, width=50, height=50)

root.mainloop()

