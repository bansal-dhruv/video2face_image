import numpy as np
# library for image
import cv2
from PIL import Image
import imutils
# path were images will be saved
path='/home/dhruv_bansal/Desktop/face/'
# haarcade given along with the file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()

    Angle=0
    maxFace=()
    count=1
    
    for angle in np.arange(0, 360, 90):
        rotated = imutils.rotate_bound(img, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(faces)
        if len(faces)>len(maxFace):
            Angle=angle
            maxFace=faces

    img = imutils.rotate_bound(img, Angle)
    # converting to gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # intializing
    faces = maxFace
    filter_img=img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        filter_img = img[y:y+h, x:x+w]
        cv2.imshow('filter', filter_img)
        cv2.imwrite(path+str(count) + ".jpg", filter_img)

        count+=1;
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    #cv2.imshow('filter', filter_img)

    k = cv2.waitKey(1)
    # press Esc key to exit
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
