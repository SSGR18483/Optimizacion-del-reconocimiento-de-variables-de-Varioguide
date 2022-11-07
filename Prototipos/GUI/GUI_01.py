# UVG
# Prototipo de GUI para el diseño final
# Santiago Galicia
# Carnet 18483


## Importe de librerias
import numpy
from PIL import ImageTk
from PIL import Image
import PIL as PIL
from tkinter import *
from tkinter.ttk import *
import os
root = Tk()
root.geometry("1000x750")

img=PIL.Image.open("D:/Documentos/UVG/QUINTO AÑO/Segundo Semestre/Diseño e innovación/GIT/Optimizacion-del-reconocimiento-de-variables-de-Varioguide/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
#img=PIL.Image.open("C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
bg = ImageTk.PhotoImage(img)

#funciones de los clicks
def exitclick():
    root.destroy()

# Create Canvas
canvas1 = Canvas(root, width=1000,height=750)
canvas1.pack(fill="both", expand=True)
# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")
# Add Text
canvas1.create_text(500, 25, text="Sistema OCR HUMANA ",font =("Courier", 30,'bold'),fill="black")

# Create Buttons

button1 = Button(root, text="Capturar Valor")
button1.place(x=50,y=50,width=100,height=50)
button2 = Button(root, text="Enviar Valores")
button2.place(x=450,y=50,width=100,height=50)
button3 = Button(root, text="Exit",command=exitclick)
button3.place(x=850,y=50,width=100,height=50)
# Display Buttons
#button1_canvas = canvas1.create_window(100, 10,
#                                       anchor="nw",
#                                       window=button1)

root.mainloop()




