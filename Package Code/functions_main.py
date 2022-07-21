#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:40:04 2022

@author: joe

Main functions which are used as preliminary processing and doesn't depend on predictors
"""

import os
from copy import deepcopy as dc
import pandas as pd
from os import path as op
import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from datetime import date
import time

# Get prediction for emotion types
def predict_emotion(imgtens,predictor):
  hipster_emote_preds = predictor.model.predict(imgtens)
  return pd.DataFrame(data=hipster_emote_preds,columns=["prob_peaceful","prob_happy","prob_gloomy","prob_scary"])


# Get predictions for media types
def predict_media(imgtens,predictor):
  hipster_media_preds = predictor.model.predict(imgtens)
  return pd.DataFrame(data=hipster_media_preds,columns=["prob_pen","prob_oilpaint","prob_watercolor","prob_comic","prob_graphite","prob_vectorart","prob_3d"])

# Compare an image output to known clothing styles and style predictions
def compare_to_styles(image_style_preds,true_styles,predicted_styles=False,pred_weight = 0.5):
  sim = 0.0
  for i in range(len(image_style_preds)):
    sim += abs(image_style_preds[i]-float(true_styles[i]))
    # If the predictions of this clothing image is known, add the differences here. If not, add the 'pred_weight'
    if(type(predicted_styles)!=bool):
      sim += pred_weight * abs(image_style_preds[i]-float(predicted_styles[i]))
    else:
      sim += pred_weight
  return sim

## Take HipsterWars predictions and combine them into single tensors
def combine_preds(types=["emote","media"],framelocs = {"emote":"emotion_predictions.csv","media":"media_predictions.csv"}):
  frames = []
  for frameloc in types:
    frames.append(pd.read_csv(framelocs[frameloc]))
  imdirs = frames[0]["image_directory"]
  outs = []
  for frame in frames:
    for col in frame.columns:
      if(col=="image_directory"):
        continue
      #print(f"Column {col} being added")
      outs.append(list(frame[col]))
  # Invert the lists obtained
  outputs = [list(i) for i in zip(*outs)]
  del outs
  # Convert the outputs to tensors
  for i in range(len(outputs)):
    outputs[i] = tf.convert_to_tensor(outputs[i],dtype=float)
  # Convert to dataframe
  return pd.DataFrame(data=list(zip(imdirs,outputs)),columns = ["image_directory","output"])

## Create a dataframe with feature outputs and clothing styles
def get_hipsterdata(attribute_types = ["emote","media"]):
  # Get the image directories and CNN outputs
  data = combine_preds(attribute_types)
  # Add the image ids and styles
  clothing_csv = pd.read_csv(open("hipster_to_csv_test.csv","r"))
  # Convert the directories to ids
  dirids = list(data["image_directory"])
  ids,nums = [],{}
  for i in range(10):
    nums[str(i)] = True
  for dirid in dirids:
    # Get the first number in the string
    numdec = len(dirid)-5
    while(dirid[numdec-1] in nums):
      numdec -= 1
    # Append the id only
    ids.append(int(dirid[numdec:-4]))
  # Add the IDs to the original data
  data["ID"] = ids
  del data["image_directory"]
  # Get the styles for each value
  styles = []
  styledict = {"Hipster":0,"Goth":1,"Preppy":2,"Pinup":3,"Bohemian":4}
  for tempid in ids:
    temp = clothing_csv.loc[clothing_csv["ID"]==tempid]
    label = list(temp["Label"])[0]
    labellist = [0 for i in range(len(styledict))]
    labellist[styledict[label]] += 1
    label = tf.convert_to_tensor(labellist,dtype=float)
    styles.append(label)
  data["style"] = styles
  return data


# Convert a series of image directories to tensors
def dirs_to_tensors(dirlist):
  imglist = [load_img(imdir) for imdir in dirlist]
  tenlist = [tf.convert_to_tensor(x) for x in imglist]
  del imglist
  inten = tf.convert_to_tensor(tenlist)
  del tenlist
  return inten

# Try training with the specified fraction of training the whole model vs part of it
def train_custom(model,inten,outten,epochs=50,all_to_fine=0.5,order=["fine","all","fine"],lrate=[1e-3,1e-4,1e-4],specifier=""):
  # Create the epoch distribution for training, constructing a guide which can easily be followed
  all_frac,fine_frac = all_to_fine/(1+all_to_fine),1/(1+all_to_fine)
  all_epochs,fine_epochs = int(round(all_frac*epochs)),int(round(fine_frac*epochs))
  all_epochs /= max(1,order.count("all"))
  fine_epochs /= max(1,order.count("fine"))
  training_guide = [[train_mode,{"all":int(all_epochs),"fine":int(fine_epochs)}[train_mode]] for train_mode in order]
  for i in range(len(training_guide)): training_guide[i].append(i+1)
  for i in range(len(training_guide)): training_guide[i].append(lrate[i])
  print(f"Training guide: {training_guide}")
  # Compile the model ready for training
  totrain = dc(model)
  print("Compiling model")
  totrain.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=lrate[0]),loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
  print(f"Number of layers in model: {len(totrain.layers)}")
  full_history = []
  # Run through the training steps and train the model appropriately
  for step in training_guide:
    # Modify trainable parameters
    if(step[0]=="all"):
      totrain.layers[0].trainable = True
    elif(step[0]=="fine"):
      totrain.layers[0].trainable = False
    # Modify the learning rate
    print(f"Current lrate: {totrain.optimizer.learning_rate}")
    print(f"New lrate: {step[3]}")
    tf.keras.backend.set_value(totrain.optimizer.learning_rate,step[3])
    # Print out progress and train the model
    print(f"Fitting step {step[2]}")
    history = totrain.fit(x=inten,y=outten,epochs=step[1],validation_split=0.2)
    full_history.append(dc(history))
    plot_history(history,f"Performance in training step {step[2]}")
    # Save the trained model
    print("Saving model")
    save_current_model(totrain,step,specifier = f"{specifier},epochs {epochs},all_to_fine {all_to_fine},order {str(order)},lrate {lrate},Step {step[2]}")
    print("Model saved")
  plot_history_multi(full_history,title="Full training history")
  return

# Save a model to the correct location
def save_current_model(model,step,specifier=""):
  # Copy the model
  tosave = dc(model)
  # Navigate to the models directory
  ddir = get_ddir()
  ddir = op.join(ddir,"models")
  # Get the date
  cur_date = str(date.today())
  # Find and make the final path
  prev_dir = ddir
  ddir = op.join(ddir,cur_date)
  if(cur_date not in os.listdir(prev_dir)):
    os.mkdir(ddir)
  del prev_dir
  # Construct the base model name
  if(specifier == ""):
    filename = cur_date
  else:
    filename = specifier
  # Check if other models have been trained this day; modify if so
  if(filename in os.listdir(ddir)):
    filename += ";2"
    while(filename in os.listdir(ddir)):
      filename[-1] = str(int(filename[-1])+1)
  # Save this model
  savedir = op.join(ddir,filename)
  tosave.save(savedir)
  return


# Plot the progression of a classifier
def plot_history(history,title=""):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if(title==""):
        plt.title('model loss')
    else:
        plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    return

# Plot multiple histories chained together
def plot_history_multi(histories,title=""):
    loss_history,val_history = [],[]
    for history in histories:
        loss_history += history.history["loss"]
        val_history += history.history["val_loss"]
    plt.plot([i for i in range(len(loss_history))],loss_history,label="Training Loss")
    plt.plot([i for i in range(len(val_history))],val_history,label="Validation Loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    if(title!=""):
        plt.title(title)
    plt.show()
    return

# Convert a dataframe to an input and output list
def convert_output(df,DEBUG=False,attribute_types=["emotion","media"]):
    ins,outs = [],[]
    # Dictionary for the different attributes
    all_attributes = {"emotion":["peaceful","happy","gloomy","scary"],"media":["pen","oilpaint","watercolor","comic","graphite","vectorart","3d"]}
    adict = {}
    for at in attribute_types:
      for a in all_attributes[at]:
        adict[a] = len(adict)
    # Dictionary for different attribute type boundaries
    limdict = {}
    for at in attribute_types:
      limdict[at] = [adict[all_attributes[at][0]],adict[all_attributes[at][-1]]]
    for mid in df["mid"].unique():
        # Append the image directory to inputs
        ins.append(img_id_to_dir(mid,DEBUG))
        rel = df.loc[df["mid"]==mid]
        # Create a new array of 0's to append to the outputs
        new = [0 for i in range(len(adict))]
        for attribute in rel["attribute"]:
            new[adict[attribute]] = 1
        # Run through and ensure if none of a given attribute type are listed they are replaced with NaN
        for lim in limdict:
          pres = False
          for i in range(limdict[lim][0],limdict[lim][1]+1):
            if(new[i]>0):
              pres = True
          if(pres==False):
            for i in range(limdict[lim][0],limdict[lim][1]+1):
              new[i] = "NaN"
        # Append the new output
        outs.append(np.array(new,dtype=float))
    return ins,outs

# Open the prepared crowd training data which doesn't need further modification (beyond converting directories to images)
def open_crowdtraining(DEBUG=False):
  ddir = get_ddir()
  data = pd.read_csv(os.path.join(ddir,"crowd_labels-training.csv"))
  data["output"] = data["output"].apply(lambda x: np.array(x))
  return training_split(data)

# Get a dataframe of directories to images and crowd labels
def get_crowdlabels():
    # Get directory
    ddir = anno_csv_dir()
    ddir = op.join(ddir,"crowd_labels.csv")
    # Convert to dataframe
    df = pd.read_csv(ddir)
    dirs = [img_id_to_dir(imgid) for imgid in df["mid"]]
    df["directory"] = dirs
    return df

# Get a training and testing set for images and crowd labels
def get_crowdtraining(DEBUG=False,attribute_types=["emotion","media"]):
    # Get the improved crowd label list
    df = pd.read_csv(os.path.join(get_ddir(),"crowd_labels-improved.csv"))
    # Remove the unwanted attributes
    for unwanted_type in ["content","emotion","media"]:
      if(unwanted_type not in attribute_types):
        df = df.loc[df["attribute-type"]!=unwanted_type]
    # Refine to positive crowd labels
    df = df.loc[df["label"]=="positive"]
    # If debug is true, refine to only columns in the prototyping folder
    if(DEBUG==True):
      print(len(df))
      proto_ids = {}
      proto_dir = get_ddir();proto_dir = os.path.join(proto_dir,"images-annotated-prototyping")
      for folder in os.listdir(proto_dir):
        for filename in os.listdir(os.path.join(proto_dir,folder)):
          if(filename[0:-4] not in proto_ids):
            proto_ids[int(filename[0:-4])] = True
      df = df[df["mid"].isin(proto_ids)]
      print(len(df))
    # Convert to appropriate datatype
    ins,outs = convert_output(df,DEBUG,attribute_types=attribute_types)
    df = pd.DataFrame(data=list(zip(ins,outs)),columns=["directory","output"])
    # Split them into test and train
    return training_split(df)

# Split crowd attributes into type and specific
def split_attributes(df):
    ref = df.copy()
    # Splite attribute into attribute type and attribute
    ref.insert(1,"attribute-type",[a.split("_")[0] for a in list(ref["attribute"])].copy())
    temp = [a.split("_")[1] for a in list(ref["attribute"])].copy()
    ref["attribute"] = temp.copy()
    return ref

# Split a dataframe into training and testing
def training_split(df,train_frac=0.8,state=2183):
    train = df.sample(frac=train_frac,random_state=state)
    dfi,traini = df.index,train.index
    mask = ~dfi.isin(traini)
    return train,df.loc[mask]

# Get the directory of the csv labels folder
def anno_csv_dir():
    ddir = os.getcwd();ddir = op.dirname(ddir);ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data");ddir = op.join(ddir,"BAM");ddir = op.join(ddir,"labels-csv")
    return ddir

# Create our extended version of resnet50
# This results in 11 outputs default (for the 4 emotions and 7 media types)
def get_rn50_extended(outs=11,activator="softmax"):
    rn50 = load_rn50()
    extended = tf.keras.models.Sequential()
    extended.add(rn50)
    extended.add(tf.keras.layers.Flatten())
    try:
        extended.add(tf.keras.layers.Dense(outs,activation=activator))
    except:
        print("Unknown final layer details, defaulting to 11 softmax neurons")
        extended.add(tf.keras.layers.Dense(11,activation="softmax"))
    return extended

# Get an imagenet-trained ResNet50 model
def load_rn50():
    return tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224,224,3),
        pooling=None,
        classes=1000)

# Open an image from its id
def open_bamimage(imgid):
    ddir = img_id_to_dir(str(imgid))
    return load_img(ddir)

# Convert an image id to a directory
def img_id_to_dir(imgid,DEBUG=False):
    strid = imgid if type(imgid)==str else str(imgid)
    # Get the folder name from the id
    folder = strid[-4:]
    # Navigate to the data directory
    ddir = get_ddir()
    if(DEBUG==False):
      ddir = op.join(ddir,"images-annotated/images-annotated")
    else:
      ddir = op.join(ddir,"images-annotated-prototyping")
    # Get the image folder and directory
    ddir = op.join(ddir,folder)
    ddir = op.join(ddir,strid+".jpg")
    return ddir

# Convert an image directory to its id
def img_dir_to_id(imgdir):
  # Get a dictionary of numbers
  numdict = {}
  for i in range(10):
    numdict[str(i)] = True
  # Get the first number present in reverse order
  lastnum = len(imgdir)
  for char in imgdir[::-1]:
    lastnum -= 1
    if(char=="."):
      break
  # Add numbers in reverse order and break when a non-number shows up
  imid = ""
  for char in imgdir[:lastnum][::-1]:
    if(char not in numdict):
      break
    imid = char + imid
  return int(imid)

# Predict a single image
def pred_img(img,model):
    imgtens = tf.convert_to_tensor([img],dtype=float)
    return model.predict(imgtens)
    """
    try:
        return model.predict(img)
    except:
        return model.signatures['predict'](imgtens)["probabilities"]
    """

# Load image, taken from BAM attributions.py
def load_img(fdir,model_shape=(224,224)):
  img = Image.open(tf.io.gfile.GFile(fdir, 'rb'))
  img = preprocess_img(img,model_shape)
  return img

# Preprocessing single image; main part is resizing the image to 224x224x3
def preprocess_img(img,model_shape=(224,224),_means=[123.68, 116.78, 103.94]):
    img = img.convert('RGB').resize(model_shape, Image.BILINEAR)
    channel_means = np.expand_dims(np.expand_dims(_means, 0), 0)
    img_arr = np.array(img, dtype=np.float32) - channel_means.astype(np.float32)
    return img_arr

# Get an imagenet-trained ResNet50 model
def get_rn50():
    return tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224,224,3),
        pooling=None,
        classes=1000)

# Get directory to data
def get_ddir():
  return ""

# Load the pre-trained BAM model
def load_bam_model(choice="obj"):
    ddir = get_bammaster_dir()
    ddir = op.join(ddir,"models")
    ddir = op.join(ddir,choice)
    return tf.saved_model.load(ddir)

# Get the directory of the bam-master folder
def get_bammaster_dir():
  return ""