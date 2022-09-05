import serial
import time
size = 5

c=0

a=0
b=0
buffer = []
puerto="COM3"
baudios = 115200

ser = serial.Serial("COM4",115200)
if  ser.isOpen():
    ser.close()
if not ser.isOpen():
        ser.open()
datos = ser.read()
print (datos)
if (datos == str('O').encode('ascii')):
    buffer.append(datos)
    for i in range(1,2):
          val = ser.read()
          buffer.append(val)
    ok = str('').join(buffer)
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
ser.close()
time.sleep(1)    


   
    
  #def showMenu():
 #     if a==1:
 #         print('hola')
         
         
 #    else:
 #        print('adios')
        
     
 #   while True:
 #       showMenu()
 