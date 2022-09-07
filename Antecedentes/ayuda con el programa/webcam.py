import cv2, time

video=cv2.VideoCapture(0)
while  True:
   
    check, frame =video.read()
    ret, frame =video.read()
    print(check)
    print (frame)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capturing",gray)
    cv2.imshow('frame', frame)
    key=cv2.waitKey(1)
  
  
    if (key==ord('q')):
        for i in range(10):
            return_value, image =video.read()
            cv2.imwrite('opencv'+str(i)+'.png', image)
            del(video)   
          
video.release()


