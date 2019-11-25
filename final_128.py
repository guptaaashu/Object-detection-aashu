# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 05:32:01 2019

@author: dell
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

train = pd.read_csv('training_set.csv')
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('images/'+train['image_name'][i], target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
Y= train.iloc[:,1:5].values
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1),copy= True)
Y=sc.fit_transform(Y)

def bb_intersection_over_union(y_true, y_pred):
    
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = tf.maximum(y_true[0], y_pred[0])
    yA = tf.maximum(y_true[1], y_pred[1])
    xB = tf.minimum(y_true[2], y_pred[2])
    yB = tf.minimum(y_true[3], y_pred[3])
    tf.cast(xA,tf.float32)
    tf.cast(yA,tf.float32)
    tf.cast(xB,tf.float32)
    tf.cast(yB,tf.float32)
 
	# compute the area of intersection rectangle
    interArea = tf.maximum(0.00, xB - xA + 1) * tf.maximum(0.00, yB - yA + 1)
    tf.cast(interArea,tf.float32)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
    y_trueArea = (y_true[2] - y_true[0] + 1) * (y_true[3] - y_true[1] + 1)
    y_predArea = (y_pred[2] - y_pred[0] + 1) * (y_pred[3] - y_pred[1] + 1)
    tf.cast(y_trueArea,tf.float32)
    tf.cast(y_predArea,tf.float32)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / (y_trueArea + y_predArea - interArea)
    tf.cast(iou,tf.float32)
 
	# return the intersection over union value
    return (1-iou)



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss=bb_intersection_over_union,optimizer='sgd',metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
model.fit(X_train,y_train,batch_size=4, epochs=8)
prediction = model.predict(X_test)
pre2= sc.inverse_transform(prediction)
y_test1=sc.inverse_transform(y_test)
test = pd.read_csv('test.csv')


test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('images/'+test['image_name'][i], target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test1 = np.array(test_image)
prediction1 = model.predict(test1)
pre1=sc.inverse_transform(prediction1)
pre1=pd.DataFrame(pre1)
test4= pd.DataFrame(test['image_name'])
test2=test4.join(pre1)

test2.to_csv('test.csv', header=['image_name','x1','x2','y1','y2'], index=False)