# UVG
# Envio wifi
# Enviar estos datos
# 1. Enviar Numero de Junta
# 2. Enviar signo
# 3. Enviar float de valor de ajuste


import serial
import time
import socket



HOST = "192.168.56.1"  # Standard loopback interface address (localhost)
PORT = 80  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)

def TCPsend(j1,s,No):
    try:
        sock = socket.socket()
        HOST = "192.168.0.1"# Ip en network
        PORT = 80# Puerto al que se envia
        sock.connect((HOST, PORT))
        mensaje = "{"+f"J1:{j1}, S:{s}, No: {No}"+"}"
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

j1 = float(1.0)
s=float(1.0)
No=float(0.2)
TCPsend(j1,s,No)