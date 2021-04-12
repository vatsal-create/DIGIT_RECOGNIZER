#IMPORTING ALL THE NECESSARY LIBRARIES REQUIRED FOR THIS PROJECT
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

np.random.seed(2)

train=pd.read_csv('train.csv') #loading the training data into train dataframe using pd.read_csv
test=pd.read_csv('test.csv') #loading the test data into the test data frame using pd.read_csv

#print(train.columns)
#print(test.columns)
X_train=train.iloc[:,1:] #extracting all the columns except the labels column and making it the features dataframe
Y_train=train['label'] #storing the labels column into the as the Y_train dataframe
#print(X_train.columns)

X_train=X_train/255.0 #so that all the pixels are between 0 to 255
test=test/255.0

X_train = X_train.values.reshape(-1, 28, 28, 1) # reshaping the datframe so as to accomodate multiple training instances under one dataframe
test = test.values.reshape(-1, 28, 28, 1)
print(X_train.shape)

Y_train=to_categorical(Y_train,num_classes=10) #converting the numerical labels into catrgorcal type values in which the corresponding label's index is set to 1, rest to 0
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.1,random_state=2) #splitiing the training data into two sets, a training and a validation set
print(X_train.shape)

# Buildding the model

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #defining the optimizer function

model.compile(optimizer=optimizer,loss="categorical_crossentropy", metrics=["accuracy"]) #compiliing the model built above
model.fit(X_train,Y_train,epochs=30,batch_size=86,validation_data=(X_val,Y_val),verbose=2) #fiting the model into the training data

score=model.evaluate(X_val,Y_val) #evaluating the model for the validation set
print("Validation loss=",score[0])
print("Validation accuracy=",score[1])

Y=model.predict(test) #prediciting the digit from the test data
print(Y.shape)

results = np.argmax(Y,axis = 1) #conveting the categorical type data into a numebrical number using np.argmax() function
results = pd.Series(results,name="Label") #making results a pandas series object froom a normal numpy object

My_submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1) #storing my prediction of the test data into a csv file
My_submission.to_csv("submission.csv",index=False)
My_submission.head() #printing the first 5 rows from my predictions

model_json=model.to_json() #converting the model to a json object for later use
with open("model.json","w") as json_file:
    json_file.write(model_json) #saving the json model into a json file

model.save_weights("model.h5") #saving weights into an h5 type file, so that the weights are easily available for later use
print("Weights successfully loaded to disk")