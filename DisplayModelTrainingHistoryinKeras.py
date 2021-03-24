# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:23:31 2021

@author: omar
"""
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


df_iris = pd.read_csv('https://data.heatonresearch.com/data/t81-558/iris.csv', 
                na_values =[ 'NA', '?'])
df_iris.head()

X = df_iris.iloc[:, 0:4].values
y = pd.get_dummies(df_iris['species']).values
cesitler = pd.get_dummies(df_iris['species']).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

model_iris = Sequential()
model_iris.add(Dense(50, input_dim=X.shape[1], activation='relu'))
model_iris.add(Dense(25, activation='relu'))
model_iris.add(Dense(y.shape[1], activation='softmax'))

model_iris.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

#monitoring = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1,
                       #   mode='auto', restore_best_weights=True)

history = model_iris.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, 
               epochs=1000)


#from sklearn.metrics import accuracy_score
#import numpy as np


#prediction = model_iris.predict(X_test)
#prediction_classes = np.argmax(prediction, axis=1)
#real_classes = np.argmax(y_test, axis=1)
#correction_rate = accuracy_score(real_classes, prediction_classes)
#print(correction_rate)


#print(history.history.keys())

keys = history.history.keys()
print(keys)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


















