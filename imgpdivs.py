
import tensorflow as tf
print("my version is : " , tf.__version__)
from tensorflow.keras.datasets import mnist
(x_train , y_train) , (x_test , y_test) = mnist.load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
from matplotlib import pyplot as plt
from jedi.api.refactoring import inline
matplotlib = inline
plt.imshow(x_train[0] , cmap='binary') #displaying an example from train data
plt.show()

y_train[0]
print (set(y_train))

from tensorflow.keras.utils import to_categorical #encoding labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
#validated_shapes
print('y_train_encoded shape:', y_train_encoded.shape)
print('y_test_encoded shape:', y_test_encoded.shape)

import numpy as np

x_train_reshaped = np.reshape(x_train , (60000 ,784))
x_test_reshaped = np.reshape(x_test , (10000 ,784))

print('x_train_reshaped shape:' , x_train_reshaped.shape)
print('x_test_reshaped shape:' , x_test_reshaped.shape)

print(set(x_train_reshaped[0]))

#dta-noramlistion
x_mean=np.mean(x_train_reshaped)
x_std=np.std(x_train_reshaped)

epsilon = 0.005

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon) 
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon) #i m not calculating its mean sepratly to reduce anamoly

#display normalised pixel value

print(set(x_train_norm[0]))

#creation of model by divya shukla

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#i am not defining the input layer because of sequential function it alredy has input in from of examples
#deination of secret layers: ~ divya

model = Sequential([
    Dense(128 ,activation='relu' , input_shape=(784,)), 
     Dense(128 ,activation='relu'),
    #output layer
     Dense(10 ,activation='softmax'
    
])
    
#compiling the model
    model.compile(
    optimizer = 'sgd' , 
    loss='categorical_crossentropy' , 
    metrics = ['accuracy']
    )
    model.summary()

#training the model~ divya shukla

model.fit(x_train_norm , y_train_encoded , epochs=3)

#evaluating the model ~ divya shukla

_ , accuracy = model.evaluate(x_test_norm , y_test_encoded)
print('test set accuracy:' , accuarcy*100 )
#predictions

preds = model.predict(x_test_norm)
print('shape of preds: ' , preds.shape)

#plotting result

plt.figure(figsize=(12,12))

start_index = 0 
for i in range(25):
    plt.subplot(5 , 5 , i=1)
    plt.grid(False)
    plt.xticks([])
    plt.yicks([])
        
    pred=np.argmax(preds[start_index+i])
    gt=y_test[start_index+i]
    
    col='g'
    if pred!=gt:
        col='r'
    plt.xlabel('i={} , pred={} , gt={}'.format(start_index+i , pred , gt))
    plt.imshow(x_test[start_index+i] , cmap = 'binary'
#predictions

preds = model.predict(x_test_norm)
print('shape of preds: ' , preds.shape)

#plotting result

plt.figure(figsize=(12,12))

start_index = 0 
for i in range(25):
    plt.subplot(5 , 5 , i=1)
    plt.grid(False)
    plt.xticks([])
    plt.yicks([])
        
    pred=np.argmax(preds[start_index+i])
    gt=y_test[start_index+i]
    
    col='g'
    if pred!=gt:
        col='r'
    plt.xlabel('i={} , pred={} , gt={}'.format(start_index+i , pred , gt))
    plt.imshow(x_test[start_index+i] , cmap = 'binary'

plt.show()

plt.plot(preds[8])
plt.show()