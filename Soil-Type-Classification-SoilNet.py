#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow==2.0.0')
get_ipython().system('pip install imutils')
get_ipython().system('pip install python-telegram-bot')

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dropout, MaxPooling2D, AveragePooling2D, Dense, Flatten, Input, Conv2D, add, Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential , Model , load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array , ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from PIL import Image
import matplotlib.pyplot as plt

import cv2
from imutils import paths
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")


# In[2]:


print("Tensorflow version: ",tf.__version__)


# # **Creating class for Telegram Bot Message Updates along with Graph**
# 
# Get access_token from Telegram app: 
# * Open Telegram mobile app 
# * Search for "BotFather" 
# * Send "/start"
# * After reply from BotFather send "/newbot"
# * Give name to your Bot Eg: Example_Bot
# * Give username to your Bot Eg: My_Example_bot *Note: User name must end with '_bot'* 
# * Done...Congratulations You have crated your own Telegram bot. Now you will get Token to access the HTTP API. Copy that Token Key.

# In[3]:


import requests
import tensorflow as tf

import tensorflow.keras.utils as np_utils

access_token = 'ACCESS TOKEN' #Copy and paste Token Key given by BotFather

class botCallback(tf.keras.callbacks.Callback):
    def __init__(self,access_token):
        self.access_token = access_token
        self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/getUpdates'
        self.response = requests.get(self.ping_url).json()
        #print(self.response)
        self.chat_id = self.response['result'][0]['message']['chat']['id']
        #self.chat_id = self.response['result']

    def send_message(self,message):
        #print('sending message')
        self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/sendMessage?'+                        'chat_id='+str(self.chat_id)+                        '&parse_mode=Markdown'+                        '&text='+message
        self.response = requests.get(self.ping_url)
    
    def send_photo(self,filepath):
        imagefile= open(filepath,"rb")
        file_dict = {'photo':imagefile}
        self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/sendPhoto?chat_id='+str(self.chat_id)
        self.response = requests.post(self.ping_url, files = file_dict)
        imagefile.close()

    def on_train_batch_begin(self, batch, logs=None):
        pass
    
    def on_train_batch_end(self, batch, logs=None):
        message = ' Iteration/Batch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(batch,logs['accuracy'],logs['loss'])
        #print(logs)
        try:
            message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
            self.send_message(message)
        except:
            pass

    def on_test_batch_begin(self, batch, logs=None):
        pass
    
    def on_test_batch_end(self, batch, logs=None):
        message = ' Iteration/Batch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(batch,logs['accuracy'],logs['loss'])
        try:
            message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
            self.send_message(message)
        except:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):

        message = ' Epoch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(epoch,logs['accuracy'],logs['loss'])
        try:
            message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
            self.send_message(message)        
        except:
            pass

class Plotter(botCallback):
 def __init__(self,access_token):
    
     super().__init__(access_token)
 def on_train_begin(self,logs=None):
     self.batch = 0
     self.epoch = []
     self.train_loss = []
     self.val_loss = []
     self.train_acc = []
     self.val_acc = []
     self.fig = plt.figure(figsize=(200,100))
     self.logs = []

 def on_epoch_end(self, epoch, logs=None):
     self.logs.append(logs)
     self.epoch.append(epoch)
     self.train_loss.append(logs['loss'])
     self.val_loss.append(logs['val_loss'])
     self.train_acc.append(logs['accuracy'])
     self.val_acc.append(logs['val_accuracy'])
     f,(ax1,ax2) = plt.subplots(1,2,sharex=True)
     #clear_output(wait=True)
     ax1.plot(self.epoch, self.train_loss, label='Training Loss')
     ax1.plot(self.epoch, self.val_loss, label='Validation Loss')
     ax1.legend()
     ax2.plot(self.epoch, self.train_acc, label='Training Accuracy')
     ax2.plot(self.epoch, self.val_acc, label='Validation Accuracy')
     ax2.legend()
     plt.savefig('Accuracy and Loss plot.jpg')
     self.send_photo('Accuracy and Loss plot.jpg')


# In[4]:


data = '/kaggle/input/cotton-leaf-disease/Cotton_Leaf_Data'
train_dir = '/kaggle/input/soil-type-classification/Soil_Data/Train'
test_dir = '/kaggle/input/soil-type-classification/Soil_Data/Test'

image_size = 224


# In[5]:


batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1/255.0,
                            rotation_range=35,
                            zoom_range=0.30,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            vertical_flip= True,
                            fill_mode="nearest")

train_data = train_datagen.flow_from_directory(train_dir,
                                              target_size=(224,224),
                                              batch_size=32,
                                              class_mode="categorical")


# In[6]:


test_datagen = ImageDataGenerator(rescale = 1/255.0)

test_data = test_datagen.flow_from_directory(test_dir,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode="categorical")


# In[7]:


model = Sequential(name="SoilNet Model")
model.add(Conv2D(32,(5,5),padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D((3,3),strides=2))

model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
#model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(128,activation="relu"))
model.add(Dense(3,activation="softmax"))

optimizer = Adam(decay=1e-2,learning_rate=0.001)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()
plot_model(model,show_shapes=True)


# In[ ]:





# In[8]:


bot_callback = botCallback(access_token)
plotter = Plotter(access_token)
callback_list = [bot_callback,plotter]
start = time.time()

history = model.fit_generator(train_data,
                    steps_per_epoch = 5,
                    validation_data = test_data,
                    validation_steps = 5,
                    epochs=5,
                    callbacks=callback_list)
end = time.time()
print("Total train time: ",(end-start)/60," mins")


# In[9]:


def plot_graph(history,string):
    plt.figure(figsize=(12,8))
    plt.plot(history.history[string],label=str(string))
    plt.plot(history.history["val_"+str(string)],label="val_"+str(string))
    plt.xlabel("Epochs")
    plt.ylabel(str(string))
    plt.show()


# In[10]:


plot_graph(history,"accuracy")
plot_graph(history,"loss")


# In[11]:


model.save("Model.h5")


# In[12]:


from IPython.display import FileLink
FileLink('Model.h5')

