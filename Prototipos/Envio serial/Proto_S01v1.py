# Prototipo para enviar datos a serial
import serial
import time
import socket

def Serialsend(j1,j2,j3,j4):
    try:
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = 'COM3'#escribir al puerto
        ser.bytesize = serial.EIGHTBITS #numero de bits por bytes
        ser.parity = serial.PARITY_NONE #que no utilize bsqueda de paridad
        ser.open()
        envio="1"
        ser.write(envio.encode('ascii'))# JA:%s_JB:%s_JC:%s_JD:%s"%(j1,j2,j3,j4))
        time.sleep(0.02)
        ser.close()
        message = "Enviado adecuadamente"
    except NameError:
        message= "dato invalido"
    except:#si no se contecta al serial entonces mandar mensaje de error
        message = "No se pudo conectar el puerto"
    return print(message)

def TCPsend(j1,j2,j3,j4):
    try:
        sock = socket.socket()
        HOST = "192.168.0.1"# Ip en network
        PORT = 80# Puerto al que se envia
        sock.connect((host, port))
        mensaje = "JA:%s_JB:%s_JC:%s_JD:%s"%(j1,j2,j3,j4)#se concatena dato
        mensaje = mensaje.encode("ascii")#se configura en ascci
        sock.send(mensaje)#manda el mensaje
        data = ""
        while len(data) < len(mensaje):
            temp = sock.recv(1)
            temp = temp.decode("ascii")
            data += temp
        print(data)
        sock.close()
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
#TCPsend(j1,j2,j3,j4)

