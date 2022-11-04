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
#bg = PhotoImage(file = "")
#bg = PhotoImage(file="C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
img=PIL.Image.open("C:/Users/galic/Documents/Diseño/GIT/Trabajo-de-Graduaci-n-SG18483/Prototipos/Imagenes/Imagenes_gui/brain.jpg")
bg = ImageTk.PhotoImage(img)
#root.title("Primer intento")
#label = Label(root, text ="OCR Humana").grid(row=0,column=0)
#label2 = Label(root, text ="Trabajo de graduacion de Santiago Galicia").grid(row = 1, column = 0)

#Boton1= Button(root, text = "Reconocer Junta").grid(row = 10, column=10)

# Create Canvas
canvas1 = Canvas(root, width=400,
                 height=400)

canvas1.pack(fill="both", expand=True)

# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")

# Add Text
canvas1.create_text(200, 250, text="Welcome")

# Create Buttons
button1 = Button(root, text="Exit")
button3 = Button(root, text="Start")
button2 = Button(root, text="Reset")

# Display Buttons
button1_canvas = canvas1.create_window(100, 10,
                                       anchor="nw",
                                       window=button1)

button2_canvas = canvas1.create_window(100, 40,
                                       anchor="nw",
                                       window=button2)

button3_canvas = canvas1.create_window(100, 70, anchor="nw",
                                       window=button3)
root.mainloop()


