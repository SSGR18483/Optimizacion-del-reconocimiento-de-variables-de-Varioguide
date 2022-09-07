# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:38:15 2021

@author: Peter
"""

import serial, time
import struct

contador = bytes(111)
indice = 0
#buf[indice] = 0
pasos = 100
buffer = []
recibiendo = False

port_arduino = 'COM5'
port_tiva = 'COM4'

ser  = serial.Serial(port=port_tiva,baudrate=115200)
ser.flushInput()

def read_until(numero_datos):
    for val in range(numero_datos):
        buffer[val] = ser.read(val)
    print(buffer)

try:
    while (ser.isOpen()):
        #if not ser.isOpen():
        #    ser.open()
        ser.write(b'b')
        #ser.write(b'S+321')
        ser.write(b'S+')
        ser.write(b'321')
        if (pasos > 100):
            ser.write(str(pasos).encode('ascii'))
        elif (pasos < 100 and pasos > 9):
            ser.write(b'0')
            ser.write(str(pasos).encode('ascii'))
        elif (pasos < 10 and pasos >= 0):
            ser.write(b'00')
            ser.write(str(pasos).encode('ascii'))
            '''
            if (pasos > 0):
                ser.write(str(pasos).encode('ascii'))
            elif (pasos < 1 or pasos == 0):
                ser.write(b'0')
                pasos = 0
            '''
        '''
        ser.write(b'S')
        ser.write(b'+')
        ser.write(b'2')
        ser.write(b'3')
        ser.write(b'4')
        '''
        ser.write(b'Z')
        
        #if (ser.read() >= str('a').encode('ascii')):
                
        '''
        for val in ser.read():
            val = ser.read()
            buffer.append(val)
            print(buffer)
            if val == 'Z':
                comando = ''.join(buffer)
                print(comando)
                #print(''.join(buffer))
                buffer = []
        '''    
        datos = ser.read()
        if (datos == str('O').encode('ascii')):
            buffer.append(datos)
            for i in range(1,2):
                val = ser.read()
                buffer.append(val)
            ok = b''.join(buffer)
            print(ok)
            buffer = []
        if (ok == b'OK'):
            print("Handshake exitoso")
                    
        #recv = ser.read(2)
        #print(recv)
        '''
        while (ser.inWaiting() > 0):
            data = ser.read(ser.inWaiting())
            print(data)
        '''
        #ser.close()
        
        if (pasos > 0):
            pasos = pasos - 1
        elif (pasos == 0):
            ser.write(b'0')
            ser.close()
        time.sleep(0.02)
        
        
except KeyboardInterrupt:
    ser.close()
finally:
    ser.close()