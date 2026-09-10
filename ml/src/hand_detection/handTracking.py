import cv2
import numpy as np
import mediapipe as mp

harcascade = cv2.
hand_cascade = cv2.CascadeClassifier("classifier_model/cascade.xml")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    success, img = cap.read()

    #writing text on the image
    font = cv2.FONT_HERSHEY_COMPLEX
    #img = cv2.putText(img, 'TS is great', (200, 200), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
    #Convert to HSV color space
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))

    #result = cv2.bitwise_and(img, img, mask = mask2)

    hand = hand_cascade.detectMultiScale(img, 1.1, 4)

    for(x, y, w, h) in hand:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 2)

    cv2.imshow("Hand", img)    
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()