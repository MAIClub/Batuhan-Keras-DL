#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:45:53 2019

@author: batuhan
"""
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from keras.utils import np_utils
import keras
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.models import Sequential
#%% Getting Dataset ready and save it for later usage
data = []
labels = []
infected = os.listdir("cell_images/Parasitized/")
uninfected = os.listdir("cell_images/Uninfected/")

for imageOne in infected:
    try:
        img = cv2.imread("cell_images/Parasitized/"+imageOne)
        img_from_array = Image.fromarray(img, "RGB")
        size_image = img_from_array.resize((75,75))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("infected data loading")

for imageTwo in uninfected:
    try:
        img = cv2.imread("cell_images/Uninfected/"+imageTwo)
        img_from_array = Image.fromarray(img, "RGB")
        size_image = img_from_array.resize((75,75))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("Uninfected data loading")
        
Dataset = np.array(data)
Labels = np.array(labels)

np.save("75by75Dataset",Dataset)
np.save("75by75Labels",Labels)
#%% Loading Dataset and preparing for cnn
loaded_dataset = np.load("75by75Dataset.npy")
loaded_labels = np.load("75by75Labels.npy")

s=np.arange(loaded_dataset.shape[0])
np.random.shuffle(s)
loaded_dataset = loaded_dataset[s]
loaded_labels = loaded_labels[s]

num_classes=len(np.unique(loaded_labels))
len_data=len(loaded_dataset)

(x_train,x_test)=loaded_dataset[(int)(0.1*len_data):],loaded_dataset[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 #Normalize
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)

(y_train,y_test)=loaded_labels[(int)(0.1*len_data):],loaded_labels[:(int)(0.1*len_data)]

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
#%% Create cnn
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(75,75,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
model.summary()
#%% Fit CNN

# compile the model with loss as categorical_crossentropy and using
#adam optimizer you can test result by trying RMSProp as well as Momentum
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fit the model with min batch size as 50[can tune batch size to some factor of 2^power ] 
fitted = model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)

accuracy = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])



from keras.models import load_model
model.save('weights_for_75px.h5')

#%% prediction func
def convert_to_array(img):
    im = cv2.imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((50, 50))
    return np.array(image)
def get_cell_name(label):
    if label==0:
        return "Paracitized"
    if label==1:
        return "Uninfected"
def predict_image(file):
    model = load_model('weights.h5')
    print("Predicting Image.................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    Cell=get_cell_name(label_index)
    return Cell,"The predicted Image is a "+Cell+" with accuracy =    "+str(acc)




























