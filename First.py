#!/usr/bin/env python
# coding: utf-8

# In[26]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np


# In[27]:


facedetect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[28]:


cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
font=cv2.FONT_HERSHEY_COMPLEX


# In[29]:


model = load_model('keras_model.h5')


# In[30]:


while True:
    success,imOrignal = cap.read()
    faces = facedetect.detectMultiScale(imOrignal,1.3,5)

    for x,y,w,h in faces:
        crop_img =imOrignal[y:y+h,x:x+h]
        img= cv2.resize(crop_img, (224,224))
        img = img.reshape(1,224,224,3)
        prediction = model.predict(img)

        a = [0,1,2,3,4,5,6]
    
        #classIndex= model.predict_classes(img)
        #probabilityValue = np.amax(prediction)
        #print(probabilityValue)
        #print(type(prediction))
        #print(prediction)
        probabilityValue = np.argmax(prediction)
        print(probabilityValue)
        
        
        
        if  probabilityValue == 0:
            cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0), 2)
            cv2.rectangle(imOrignal,(x,y-40),(x+w, y),(0,255,0),-2)
            cv2.putText(imOrignal,'Angry',(x,y-10),font,0.75,(255,255,255),1,cv2.LINE_AA)
            #print("angry")
            file = open('data-of-angry.txt','a')
            file.write('angry')
            file.close()
            cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)

            """file = open("data.txt","a")
            file.write(str(probabilityValue))
            file.write("\n")
            file.close()"""



        elif probabilityValue ==1:
            cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imOrignal,'Disgust',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            #print("Disgust")
            #---------
            file = open('data-of-disgust.txt','a')
            file.write('Disgust')
            file.close()

            cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
        
        elif probabilityValue==2:
            cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imOrignal,'fear',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            #print("fear") --------------
            file = open('data-of-fear.txt','a')
            file.write('fear')
            file.close()

            cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
            
        elif probabilityValue ==3:
            cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imOrignal,'happy',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            #print("happy")

            file = open('data-of-happy.txt','a')
            file.write('happy')
            file.close()
            cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
        
        elif probabilityValue ==4:
            cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imOrignal,'neutral',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            #print("neutral")

            file = open('data-of-neutral.txt','a')
            file.write('neutral')
            file.write("\n")
            file.close()
            
            cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
        
        elif probabilityValue==5:
            cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imOrignal,'sad',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            #print("sad")
            file = open('data-of-sad.txt','a')
            file.write('sad')
            file.close()
            cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
        
        elif probabilityValue==6:
            cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv2.putText(imOrignal,'suprise',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            #print("suprise")
            file = open('data-of-suprise.txt','a')
            file.write('suprise')
            file.close()
            cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
    
        
        
    cv2.imshow("Result",imOrignal)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break


# In[31]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




