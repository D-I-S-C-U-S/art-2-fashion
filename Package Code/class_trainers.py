#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:07:53 2022

@author: joe
"""

from functions_main import *

# Class to train a model on BAM in these specific circumstances
class BAM_trainer:
  # Initialise trainer with default variables
  def __init__(self,attribute_type = "emotion",train_prop=0.1,epochs=50,all_to_fine=0.5,order=["fine","all","fine"],lrate=[1e-3]):
    self.attribute_type = attribute_type
    self.train_prop=train_prop
    if(type(train_prop)!=float):
      self.train_prop = 1.0
    else:
      while(self.train_prop > 1.0):
        self.train_prop /= 10
    self.epochs = epochs
    self.all_to_fine = all_to_fine
    self.order = order
    self.lrate = lrate
    if(type(self.lrate)!=list): self.lrate = [self.lrate]
    while(len(self.lrate)<len(self.order)):
      self.lrate.append(lrate[-1])
  
  # Alter variables for the training process
  def set_variables(self,train_prop=False,epochs=False,all_to_fine=False,order=False,lrate=False):
    if(type(train_prop)!=bool):
      self.train_prop = train_prop
    if(type(epochs)!=bool):
      self.epochs = epochs
    if(type(all_to_fine)!=bool):
      self.all_to_fine = all_to_fine
    if(type(order)!=bool):
      self.order = order
    if(type(lrate)!=bool):
      self.lrate = lrate
  
  def train(self,specifier=""):
    # Get the ResNet model
    rn = get_rn50_extended(outs={"emotion":4,"media":7}[self.attribute_type])
    # Get the training data directories and unformatted outputs
    train,test = get_crowdtraining(attribute_types=[self.attribute_type])
    print(f"{len(train)} training points available")
    # Convert the training rows into inputs and outputs for testing
    ims,outs = [],[]
    print("Loading images")
    # If 'train prop' is less than full, randomly get that much of the data
    if(self.train_prop<1.0):
      train = train.sample(frac=self.train_prop,random_state=616)
    finlen,starttime,temptime = len(train["directory"]),time.time(),time.time()
    for i in range(len(train["directory"])):
        indir,out = list(train["directory"])[i],list(train["output"])[i]
        try:
          ims.append(load_img(indir))
          outs.append(dc(out))
          if(time.time()-temptime>60):
            tottime = round((time.time()-starttime)/60,1)
            compperc = round((i/finlen)*100,1)
            print(f"{compperc}% of images loaded after {tottime} minutes")
            temptime = time.time()
        except:
          continue
    dtrain = pd.DataFrame(data=list(zip(ims,outs)),columns=["input","output"])
    # Delete the original training frame to save memory
    del train
    # Convert dtrain to something the model can train on
    intensorlist = [tf.convert_to_tensor(x) for x in dtrain["input"]]
    inten = tf.convert_to_tensor(intensorlist)
    del intensorlist
    outtensorlist = []
    for x in dtrain["output"]:
        outtensorlist.append(tf.convert_to_tensor(x,dtype=tf.float16))
    # Now that we have the input and output tensors, delete the original dtrain dataframe
    del dtrain
    outten = tf.convert_to_tensor(outtensorlist)
    print("Example out tensors:")
    for i in range(10):
      print(outten[i])
    # Train the model
    train_custom(rn,inten,outten,self.epochs,self.all_to_fine,self.order,self.lrate,specifier=specifier)
    return

# Train a hipsterwars model with specific inputs
class Hipster_trainer:
  # Initialise trainer with default variables
  def __init__(self,attribute_types = ["emote","media"],train_prop=1.0,epochs=50,all_to_fine=0.5,order=["fine","all","fine"],lrate=[1e-3]):
    self.attribute_types = attribute_types
    self.train_prop=train_prop
    if(type(train_prop)!=float):
      self.train_prop = 1.0
    else:
      while(self.train_prop > 1.0):
        self.train_prop /= 10
    self.epochs = epochs
    self.all_to_fine = all_to_fine
    self.order = order
    self.lrate = lrate
    if(type(self.lrate)!=list): self.lrate = [self.lrate]
    while(len(self.lrate)<len(self.order)):
      self.lrate.append(lrate[-1])
  
  # Alter variables for the training process
  def set_variables(self,train_prop=False,epochs=False,all_to_fine=False,order=False,lrate=False):
    if(type(train_prop)!=bool):
      self.train_prop = train_prop
    if(type(epochs)!=bool):
      self.epochs = epochs
    if(type(all_to_fine)!=bool):
      self.all_to_fine = all_to_fine
    if(type(order)!=bool):
      self.order = order
    if(type(lrate)!=bool):
      self.lrate = lrate
  
  def train(self,specifier=""):
    # Get the training data directories and unformatted outputs
    train,test = training_split(get_hipsterdata(self.attribute_types))
    print(f"{len(train)} training points available")
    # Convert the training rows into inputs and outputs for testing
    print("Loading images")
    # If 'train prop' is less than full, randomly get that much of the data
    if(self.train_prop<1.0):
      train = train.sample(frac=self.train_prop,random_state=616)
    finlen,starttime,temptime = len(train["output"]),time.time(),time.time()
    feature_inputs,outputs = [],[]
    for i in range(len(train["style"])):
        style,features = list(train["style"])[i],list(train["output"])[i]
        try:
          feature_inputs.append(dc(features))
          outputs.append(style)
          if(time.time()-temptime>60):
            tottime = round((time.time()-starttime)/60,1)
            compperc = round((i/finlen)*100,1)
            print(f"{compperc}% of images loaded after {tottime} minutes")
            temptime = time.time()
        except:
          continue
    dtrain = pd.DataFrame(data=list(zip(feature_inputs,outputs)),columns=["input","output"])
    # Delete the original training frame to save memory
    del train
    # Convert dtrain to something the model can train on
    intensorlist = [tf.convert_to_tensor(x) for x in dtrain["input"]]
    inten = tf.convert_to_tensor(intensorlist)
    del intensorlist
    outtensorlist = []
    for x in dtrain["output"]:
        outtensorlist.append(x)
    # Now that we have the input and output tensors, delete the original dtrain dataframe
    del dtrain
    outten = tf.convert_to_tensor(outtensorlist)
    print("Example out tensors:")
    for i in range(10):
      print(outten[i])
    # Create and train the model
    model_inputs = tf.keras.Input(shape=(11,))
    model_outputs = tf.keras.layers.Dense(5,activation=tf.nn.softmax)(model_inputs)
    rn = tf.keras.Model(inputs=model_inputs,outputs=model_outputs)
    train_custom(rn,inten,outten,self.epochs,self.all_to_fine,self.order,self.lrate,specifier=specifier)
    return