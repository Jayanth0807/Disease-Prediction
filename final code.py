
import os
import sys
import RPi.GPIO as  GPIO
from time import sleep
import time
import urllib.request            
import serial           
import telepot


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

ser = serial.Serial ("/dev/ttyUSB0",baudrate=9600,timeout=1)              #Open port with baud rate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, make_scorer
     
accdt = pd.read_csv("Covid_dataset.csv",encoding='cp1252')
print(accdt)
y = accdt['Covid']
X = accdt.drop(['Covid'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
knn_classifier = KNeighborsClassifier(n_neighbors = 4)
knn_classifier.fit(X_train, y_train)

knn_preds = knn_classifier.predict(X_test)
knn_acc = accuracy_score(y_test, knn_preds)
print("Accuracy with KNN: ", accuracy_score(y_test, knn_preds))


svc_clf = SVC(gamma='scale')
svc_clf.fit(X_train,y_train)
svc_preds = svc_clf.predict(X_test)
svc_acc = accuracy_score(y_test, svc_preds)
print("Accuracy with SVC: ", accuracy_score(y_test, svc_preds))

regr = LogisticRegression(solver="liblinear").fit(X_train,y_train)
regr_preds = regr.predict(X_test)
regr_acc = accuracy_score(y_test, regr_preds)
print("Accuracy with LR: ", accuracy_score(y_test, svc_preds))

data = []




def handle(msg):
  global telegramText
  global chat_id
  global receiveTelegramMessage
  
  chat_id = msg['chat']['id']
  telegramText = msg['text']
  
  print("Message received from " + str(chat_id))
  
  if telegramText == "/start":
    bot.sendMessage(chat_id, "Welcome to ENERGY MONITORING Bot")

  if telegramText == "/on":
    GPIO.output(rel,1)

  if telegramText == "/off":
    GPIO.output(rel,0)
  
  else:
    
    receiveTelegramMessage = True

bot = telepot.Bot('5698241063:AAHlqqYSFZ5zTwNYfm6BDpol7in4HU-ZXUQ')
chat_id='15171262010'
#bot.message_loop(handle)

print("Telegram bot is ready")

#bot.sendMessage(chat_id, 'BOT STARTED')
#time.sleep(2)



while(True):
    if(1): 
        received_data = ser.readline()                  #read NMEA string received
        
        if(received_data != None):
            received_data=received_data.decode()[:-2]
            received_data=received_data.replace(' /',':')
            #print(received_data)
            info=received_data.split(':')
            hb=info[1]
            sp=info[3]
            tm=info[5]
            ecg=info[7]
            #frequency=info[5]
            #pf=info[6]

            print("HB   :"+ str(hb))
            print("SPO2   :"+ str(sp))
            print("TEMP :"+ str(tm))
            print("ECG:"+ str(ecg))
            print("")



            time.sleep(0.03)
            data.append(ecg)
            plt.plot(data)
            plt.pause(0.01)
            plt.clf()
            if(float(tm)>0 and int(hb)>40 and int(sp)>40):
              inputValues = [0,0,0]  
              inputValues[0]=tm
              inputValues[1]=hb
              inputValues[2]=sp
              print(inputValues)
              final_Result = knn_classifier.predict([inputValues])
             # final_Result = svc_clf.predict([inputValues])
              print(final_Result)
              
              if final_Result[0] == 1:
                  print('Covid Symptoms Identified Please Consult Doctor')
              else: 
                  print('You Are SAFE')  
                  
    else:
            time.sleep(0.001)
        
