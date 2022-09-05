# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:04:38 2021

@author: joe
"""

import tkinter as tk
from tkinter import Tk, Label, Button
root=Tk()
#creacion de pantalla
def myClick():
    myLabel=Label(root,text="sos feo")
    myLabel.pack()
    
myButton=Button(root, text="papi rico",padx=50,pady=50,command=myClick,fg="blue",bg="#00ffff")

#myButton=Button(root, text="papi rico",padx=50,pady=50,command=myClick)
#myButton=Button(root, text="papi rico",padx=50,pady=50)
#myButton=Button(root, text="papi rico",state='disabled')
#myButton=Button(root, text="papi rico")
myButton.pack()




root.mainloop()