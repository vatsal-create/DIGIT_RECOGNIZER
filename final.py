from operator import mod
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import model_from_json
from keras.models import Sequential
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2


json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Model loaded from disk")

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

loaded_model.compile(optimizer=optimizer,loss="categorical_crossentropy", metrics=["accuracy"])

'''cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('video',gray)
    if cv2.waitKey(2)==ord(' '):
        if cv2.waitKey(0)==ord('c'):
            continue
        if cv2.waitKey(0)==ord('s'):
            print("Evaluating")
            img=gray
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img=2*img
            print(img)
            _, img = cv2.threshold(img,70, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyWindow('image')
            img = pd.DataFrame(img)
            img = img.values.reshape(1, 28, 28, 1)
            Y = loaded_model.predict(img)
            for i in range(0, Y.shape[0]):
                for j in range(0, 10):
                    if (Y[i][j] == 1):
                        print(j)
                        break

        if cv2.waitKey(0)==ord('q'):
            print("Quiting!!!")
            break
        if cv2.waitKey(0)==ord('p'):
            print(gray.shape)
            print(gray)
cap.release()
cv2.destroyAllWindows()'''

img=cv2.imread('9.jpg',0)
#img=cv2.resize(img,(28,28),cv2.INTER_AREA)
print(img.shape)
#_,img=cv2.threshold(img,130,255,cv2.THRESH_BINARY_INV)
cv2.imshow('name',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(img)

img=pd.DataFrame(img)
img=img.values.reshape(1,28,28,1)

#test=pd.read_csv('test.csv')
#test=test.values.reshape(-1,28,28,1)

Y=loaded_model.predict(img)
print(Y)
for i in range(0,Y.shape[0]):
    for j in range(0,10):
        if(Y[i][j]==1):
            print(j)
            break
#print(Y)