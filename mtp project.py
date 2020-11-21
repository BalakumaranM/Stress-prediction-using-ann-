#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from numpy import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def data_value(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["t_web", "t_uf", "t_lf"]
	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainX = cs.fit_transform(train[continuous])
	testX = cs.transform(test[continuous])
	#print(testX)
	print(trainX)   
	return (trainX, testX)




def create_mlp(dim, regress=False):
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))
	# return our model
	return model


# To load data

cols = ["t_web", "t_uf", "t_lf", "e_s"]
df = pd.read_csv(r"D:\MTP\my project\detail.csv", sep=",", header=None, names=cols)


(train, test) = train_test_split(df, test_size=0.35, random_state=1)
print(train)

EqStress = train["e_s"].max()
trainY = train["e_s"] / EqStress
testY = test["e_s"] / EqStress

print(" processing data")
(trainX, testX) = data_value(df, train, test)



model = create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)




print("[INFO] training model...")
history = model.fit(x=trainX, y=trainY,validation_data=(testX, testY),epochs=200)



# make predictions on the testing data
preds = model.predict(testX)
#print(preds)

x_in =np.array([[1,0.5,0.5 ]])
x_in.reshape(3,)
y_res=model.predict(x_in)
y_res.flatten()
print(y_res*EqStress)



diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print(history.history.keys())
#%tensorboard --logdir logs/fit


    


# In[5]:


import matplotlib.pyplot as plt
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,201)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[3]:


ann_viz(model, view=True, filename=”network.gv”, title=”MyNeural Network”)


# In[1]:


pip install ann_visualizer


# In[ ]:





# In[ ]:


loss_train = history.history['acc']
loss_val = history.history['val_acc']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


print(history.history.keys())


# In[ ]:




