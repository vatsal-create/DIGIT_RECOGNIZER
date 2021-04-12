# DIGIT_RECOGNIZER

The project is to recognise single digits from the MNIST dataset using the concepts of Deep Learning(DL) and Convulutional Neural Networks (CNN) and the concept of model saving and its usage later. This project is my submission to the online digit recognising competition held on kaggle [LINK TO THE COMPETITION](https://www.kaggle.com/c/digit-recognizer). This project provided an accuracy of 99.11% on the given test data(file added in the repo).

## MODEL USED
I used the super powerful keras deep learning library provided by python for the development of this project. This library is best suited for complex and often tedious DL tasks and hence has always been my ideal choice. The model used is a simple CNN having this assmbly : CONV(32)-CONV(32)-MAXPOOL-DROPOUT-CONV(64)-CONV(64)-MAXPOOL-DROPOUT-FLATTEN.
This model has been fitted to the train data provided by the competition and then saved as a json file(file in repo) for later use and also to avoid the delay in program execution due to fitting if the model on each run.

## SAMPLE IMAGES

![6](https://github.com/vatsal-create/DIGIT_RECOGNIZER/blob/main/6.jpg)
![5](https://github.com/vatsal-create/DIGIT_RECOGNIZER/blob/main/5.jpg)
![7](https://github.com/vatsal-create/DIGIT_RECOGNIZER/blob/main/7.jpg)
![8](https://github.com/vatsal-create/DIGIT_RECOGNIZER/blob/main/8.jpg)
![9](https://github.com/vatsal-create/DIGIT_RECOGNIZER/blob/main/9.jpg)
