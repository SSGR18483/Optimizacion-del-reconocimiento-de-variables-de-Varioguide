# UVG
# Prototipo de GUI para el diseño final
# Santiago Galicia
# Carnet 18483


## Importe de librerias
import numpy
from PIL import ImageTk
from PIL import Image
from PIL import ImageGrab
import PIL as PIL
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import os,time,threading
root = Tk()
root.geometry("1000x750")

img=PIL.Image.open("D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
# img=PIL.Image.open("C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
bg = ImageTk.PhotoImage(img)
cortada = PIL.Image.open("D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/OCR1/cutted.jpg")
recortetk = ImageTk.PhotoImage(cortada)

Joint = 1
ADJ1 = 0.2
ADJ2 = 0
#funciones de los clicks
#salir
def exitclick():
    root.destroy()
#limpiar
def cleart():
    mytext1.delete(1.0,END)

def mostrarJ():
    if Joint == 1 or Joint == 3:
        oracion = f"La junta {Joint} tiene:\n-Un ajuste de grado de: {ADJ1}º"
    elif Joint == 2:
        oracion = f"La junta {Joint} tiene:\n-Un ajuste de ángulo de: {ADJ1}º \n-Un ajuste de desplazamiento de: {ADJ2}mm"
    else:
        oracion = f"No se reconoce ninguna junta o ajuste."
    mytext1.insert(0.1,str(oracion))

def showimagencort():
    imgbox = Text(root, width=25, height=5)
    imgbox.place(x=0, y=660)
    imgbox.window_create(tk.END, window=tk.Label(imgbox, image=recortetk))
# Create Canvas
canvas1 = Canvas(root, width=1000,height=750)
canvas1.pack(fill="both", expand=True)
# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")
# Add Text
canvas1.create_text(500, 25, text="Sistema OCR HUMANA ",font =("Courier", 30,'bold'),fill="black")

#crear widget de texto
mytext1 = Text(root, width= 40, height=5,font=("Tahoma",12,'bold'))
mytext1.place(x=300,y=345)


# Create Buttons


button1 = Button(root, text="Determinar tipo de Junta")
button1.place(x=25,y=50,width=150,height=50)
button2 = Button(root, text="Reconocer Valores",command=mostrarJ)
button2.place(x=440,y=50,width=130,height=50)
button3 = Button(root, text="Salir",command=exitclick)
button3.place(x=900,y=700,width=100,height=50)
button4 = Button(root, text="Enviar Valores")
button4.place(x=850,y=50,width=100,height=50)
button5 = Button(root, text="Limpiar",command= cleart)
button5.place(x=705,y=395,width=50,height=50)
button5 = Button(root, text="Mostar digito",command=showimagencort )
button5.place(x=25,y=150,width=50,height=50)

# Display Buttons
#button1_canvas = canvas1.create_window(100, 10,
#                                       anchor="nw",
#                                       window=button1)

root.mainloop()




