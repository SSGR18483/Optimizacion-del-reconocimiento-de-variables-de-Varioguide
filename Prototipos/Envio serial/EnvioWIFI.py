#
#     __  ___    ________
#    / / / / |  / / ____/
#   / / / /| | / / / __
#  / /_/ / | |/ / /_/ /
#  \____/  |___/\____/
#
#  envio tcp
#
# UVG
# Envio wifi
# Enviar estos datos
# 1. Enviar Numero de Junta
# 2. Enviar signo
# 3. Enviar float de valor de ajuste


import serial
import time
import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
HOST = "192.168.5.59"  # Standard loopback interface address (localhost)
PORT = 80  # Port to listen on (non-privileged ports are > 1023)

server_address = (HOST, PORT)
sock.bind(server_address)

juntas = [[1.1, 0.2, 1.1]]#, [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]


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


if __name__ == "__main__":
    # Listen for incoming connections
    sock.listen(1)
    j1 = 1.1
    s = 1.0
    No = 0.2

    print("Esperando conexión...")
    connection, client_address = sock.accept()

    num_juntas = len(juntas)
    for index in range(0, num_juntas):
        print("Mandar mensaje...")
        msg = TCPsend(juntas[index])
        print("Recibir mensaje...")
        msg_recv = TCPreceive()

        print(msg)
        print("\n")
        print(msg_recv)
        print("\n")
