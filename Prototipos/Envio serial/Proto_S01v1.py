# Prototipo para enviar datos a serial
import serial
import time
import socket

def Serialsend(j1,j2,j3,j4):
    try:
        ser = serial.Serial()
        ser.port = 'COM5'#escribir al puerto
        ser.baudrate = 9600
        # ser.bytesize = serial.EIGHTBITS #numero de bits por bytes
        # ser.parity = serial.PARITY_NONE #que no utilize bsqueda de paridad
        ser.write("JA:%s_JB:%s_JC:%s_JD:%s"%(j1,j2,j3,j4))
        time.sleep(0.5)
        ser.close()
        message = "Enviado adecuadamente"
    except NameError:
        message= "dato invalido"
    except:#si no se contecta al serial entonces mandar mensaje de error
        message = "No se pudo conectar el puerto"
    return print(message)

def TCPsend(j1,j2,j3,j4):
    HOST = "192.168.0.1"
    PORT = 65432
    
    try:

    except NameError:
        message='Dato invalido'
    except:
        message='No se pudo conectar al cliente'
    return print(message)
#TEST
j1=.05
j2=4.4
j3=1
j4=0.2
print("Junta A:%s,\t Junta B%s,\t Junta C:%s,\t Junta D:%s"%(j1,j2,j3,j4))
Serialsend(j1,j2,j3,j4)

