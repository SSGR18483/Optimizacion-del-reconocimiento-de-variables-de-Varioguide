# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:13:22 2021

@author: joe
"""

import tkinter as tk
from tkinter import Tk, Label, Button

root=Tk()
root.title("calculadora")

def button_add():
    return
e =tk.Entry(root,width=35,borderwidth=5)
e.grid(row=0,column=0,columnspan=3,padx=10,pady=10)
#definir botones

buttom1=Button(root,text="1",padx=40,pady=20,command=button_add)
buttom2=Button(root,text="2",padx=40,pady=20,command=button_add)
buttom3=Button(root,text="3",padx=40,pady=20,command=button_add)
buttom4=Button(root,text="4",padx=40,pady=20,command=button_add)
buttom5=Button(root,text="5",padx=40,pady=20,command=button_add)
buttom6=Button(root,text="6",padx=40,pady=20,command=button_add)
buttom7=Button(root,text="7",padx=40,pady=20,command=button_add)
buttom8=Button(root,text="8",padx=40,pady=20,command=button_add)
buttom9=Button(root,text="9",padx=40,pady=20,command=button_add)
buttom0=Button(root,text="0",padx=40,pady=20,command=button_add)
buttom_p=Button(root,text="+",padx=30,pady=20,command=button_add)
buttom_e=Button(root,text="=",padx=91,pady=20,command=button_add)
buttom_c=Button(root,text="Clear",padx=79,pady=20,command=button_add)

buttom1.grid(row=3,column=0)
buttom2.grid(row=3,column=1)
buttom3.grid(row=3,column=2)

buttom4.grid(row=2,column=0)
buttom5.grid(row=2,column=1)
buttom6.grid(row=2,column=2)

buttom7.grid(row=1,column=0)
buttom8.grid(row=1,column=1)
buttom9.grid(row=1,column=2)

buttom0.grid(row=4,column=0)
buttom_p.grid(row=4,column=1)
buttom_e.grid(row=5,column=0)
buttom_c.grid(row=4,column=1,columspan=2)
#put bottom on screen









root.mainloop()