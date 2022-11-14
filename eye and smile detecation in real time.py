import cv2
from matplotlib import pyplot as plt
import numpy as np
cap=cv2.VideoCapture(0)
faceClassifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
smileClassifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
eyeClassifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')



while (cap.isOpened()):

    ret,frame=cap.read()

    if (ret):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceClassifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2,minSize=(70,70))
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,200,0),5)
            cv2.putText(frame,"me",(x+w-150,y+h+30),5,1.7,(255,255,0),3)
            face=gray[y:y+h,x:x+w]
            eyes=eyeClassifier.detectMultiScale(face,scaleFactor=1.01,minNeighbors=5,minSize=(40,40))
            for x2,y2,w2,h2 in eyes:
                cv2.circle(frame,(int(x+x2+(w2/2)),int(y+y2+(h2/2))),20,(0,250,),3)
            
                cv2.putText(frame,"eye",(x+x2+w2-40,y+y2+h2+20),2,0.9,(255,255,0),2)
                smile=smileClassifier.detectMultiScale(face,scaleFactor=1.99,minNeighbors=5,minSize=(60,40))
                for x3,y3,w3,h3 in smile:
                    cv2.rectangle(frame,(x+x3,y+y3),(x+x3+w3,y+y3+h3),(0,255,0),5)
                    cv2.putText(frame,"smile",(int(x+x3+w3-90),int(y+y3+h3+20)),2,0.9,(0,250,250),2)

    cv2.imshow("my_image",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()