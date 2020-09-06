# Telegram-Bot-for-Model-Training-Updates :rocket:
Created a Telegram Bot that will send message after every Epoch regarding Training &amp; Validation Accuracy, Loss and image of Graph too.

# How to make it ?? :heart_eyes:
Follow below steps:-> :heavy_check_mark:

Get access_token from Telegram app: :+1:

-Open Telegram mobile app

-Search for "BotFather"

-Send "/start"

-After reply from BotFather send "/newbot"

-Give name to your Bot Eg: Example_Bot

-Give username to your Bot Eg: My_Example_bot Note: User name must end with '_bot'

-Done...Congratulations You have crated your own Telegram bot. Now you will get Token to access the HTTP API. Copy that Token Key. :bowtie:


# Code Section
Requirements...
You need to install `python-telegram-bot` in-order to create a Class which will contain functions to pass message over HTTP API. Below is the command to install telegram bot package.

```!pip install python-telegram-bot```

Now you need to add python program in your model training program.

```
import requests
import tensorflow as tf

import tensorflow.keras.utils as np_utils

access_token = 'ACCESS TOKEN' #replace ACCESS TOKEN with the Access Token given to you by BotFather.


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
        self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/sendMessage?'+\
                        'chat_id='+str(self.chat_id)+\
                        '&parse_mode=Markdown'+\
                        '&text='+message
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
```

Now create object of both classes.
```
bot_callback = botCallback(access_token)
plotter = Plotter(access_token)
callback_list = [bot_callback,plotter]

```
Pass to `model.fit(.....,callbacks=callback_list)` :boom: :star:

You can view my code for reference
[Complete Example](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/Soil-Type-Classification-SoilNet.ipynb)



# Screen Shots :relaxed:

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(163).png)

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(164).png)

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(165).png)

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(166).png)

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(167).png)

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(168).png)

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(169).png)

![alt text](https://github.com/OMIII1997/Telegram-Bot-for-Model-Training-Updates/blob/master/screen_shots/Screenshot%20(170).png)

