#!/usr/bin/env python
# coding: utf-8

# In[1]:


""""""
THIS PROGRAM IS CREATED BY Aakasha01Agarwal

PRESS ESCAPE TO QUIT

""""""


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

cap=cv.VideoCapture(0)


face_cascade=cv.CascadeClassifier("/ PATH OF THE .XNL FILE ")


def face_detect_blur(img):
    face_image=img.copy()
    roi=img.copy()
    rect=face_cascade.detectMultiScale(face_image,scaleFactor=1.2)
    for (x,y,w,h) in rect:
        cv.rectangle(face_image,(x,y),(x+w,y+h),(0,255,0),10)
        roi=roi[y:y+h,x:x+w]
        blur_roi=cv.medianBlur(roi,49)
        if blur_roi is not None:
            face_image[y:y+h,x:x+w]=blur_roi
    return face_image



while True:
    ret,frame=cap.read(0)
    frame=face_detect_blur(frame)
    frame=cv.imshow("VIDEO",frame)
    k=cv.waitKey(1)
    if k==27:
        break
cap.release()
cv.destroyAllWindows()


# In[ ]:




