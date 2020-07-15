# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script file.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow import lite
import numpy as np
#dataset using numpy
x=np.array([1,0,4,5,2],dtype=float)#feature
y=np.array([1,-1,7,9,3],dtype=float)#label

#y=2x-1 this is pattern which this model will identify
## keras already has input layers

#unit means no.of neurons in hidden layer
#inputshape means the neuron at previous output(in this case the x is one variable acts as input)

model=keras.Sequential([keras.layers.Dense(units=1,input_shape=[1]),keras.layers.Dense(units=1,input_shape=[1])])
#compiing model
#loss function->how different the predicted result from actual result
#optimizer finds the minimum of loss function .
model.compile(optimizer="sgd",loss="mean_squared_error")
#training model
model.fit(x,y,epochs=500)
#test model
print(model.predict([10]))

#result must be 19   we got [[18.999985]]  it predicted the pattern 

kearas_file = "linear.h5"
tf.keras.models.save_model(model,kearas_file)
converter = lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("linear.tflite","wb").write(tfmodel)



