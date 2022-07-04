#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:53:58 2022

@author: joe
"""

import os
from os import path as op
import tensorflow as tf
import random as rand
from PIL import Image
import numpy as np
import pandas as pd
from copy import deepcopy as dc
from matplotlib import pyplot as plt

def main():
    # Get the ResNet model
    #rn = load_rn50()
    rn = get_rn50_extended()
    # Get the training data
    train,test = open_crowdtraining()
    ### DEBUG
    # Convert the first 100 training rows into inputs and outputs for testing
    dtrain = train.iloc[0:100]
    ims,outs = [],[]
    for i in range(len(dtrain["input_directory"])):
        indir,out = list(dtrain["input_directory"])[i],list(dtrain["output"])[i]
        try:
            ims.append(load_img(indir))
            outs.append(dc(out))
        except:
            continue
    dtrain = pd.DataFrame(data=list(zip(ims,outs)),columns=["input","output"])
    # Convert dtrain to something the model can train on
    intensorlist = [tf.convert_to_tensor(x) for x in dtrain["input"]]
    inten = tf.convert_to_tensor(intensorlist)
    outtensorlist = []
    for x in dtrain["output"]:
        y = []
        for char in x:
            try:
                y.append(int(char))
            except:
                continue
        outtensorlist.append(tf.convert_to_tensor(y,dtype=tf.int8))
    outten = tf.convert_to_tensor(outtensorlist)
    test_training(rn,inten,outten,99)
    """
    # Make the base model untrainable so we can fine-tune our new layer
    rn.layers[0].trainable = False
    print("compiling")
    rn.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-3),loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
    print("fitting")
    history = rn.fit(x=inten,y=outten,epochs=100,validation_split=0.2)
    plot_history(history)
    # Make the rest of the layers trainable again
    rn.layers[0].trainable=True
    """
    ### DEBUG END
    #img = debug_load_image()
    return

# Test the difference in accuracy and loss when the whole model is trained, only new layers are trained, half and half
def test_training(model,inten,outten,epochs=50):
    alltrain = dc(model)
    print("compiling all-train")
    alltrain.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-3),loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
    print("fitting all-train")
    all_history = alltrain.fit(x=inten,y=outten,epochs=epochs,validation_split=0.2)
    plot_history(all_history,"Loss when unfrozen")
    
    finetrain = dc(model)
    finetrain.layers[0].trainable = False
    print("compiling fine-train")
    finetrain.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-3),loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
    print("fitting fine-train")
    fine_history = finetrain.fit(x=inten,y=outten,epochs=epochs,validation_split=0.2)
    plot_history(fine_history,"Loss when fine-tuning")
    
    halftrain = dc(model)
    halftrain.layers[0].trainable = False
    print("compiling half-train")
    halftrain.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-3),loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
    print("fitting half-train")
    halfhistories = []
    halftrain.layers[0].trainable = False
    halfhistories.append(halftrain.fit(x=inten,y=outten,epochs=int(epochs/3),validation_split=0.2))
    halftrain.layers[0].trainable = True
    halfhistories.append(halftrain.fit(x=inten,y=outten,epochs=int(epochs/3),validation_split=0.2))
    halftrain.layers[0].trainable = False
    halfhistories.append(halftrain.fit(x=inten,y=outten,epochs=int(epochs/3),validation_split=0.2))
    plot_history_multi(halfhistories,title="Loss when half-training")
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
def convert_output(df):
    ins,outs = [],[]
    adict = {'pen': 0, 'oilpaint': 1, 'watercolor': 2, 'comic': 3, 'graphite': 4, 'vectorart': 5,
             '3d': 6, 'peaceful': 7, 'happy': 8, 'gloomy': 9, 'scary': 10}
    for mid in df["mid"].unique():
        ins.append(img_id_to_dir(mid))
        rel = df.loc[df["mid"]==mid]
        new = [0 for i in range(len(adict))]
        for attribute in rel["attribute"]:
            new[adict[attribute]] = 1
        outs.append(np.array(new,dtype=int))
    return ins,outs

# Open the prepared crowd training data which doesn't need further modification (beyond converting directories to images)
def open_crowdtraining():
    data = pd.read_csv("crowd_labels-training.csv")
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
def get_crowdtraining():
    # Get the improved crowd label list
    df = pd.read_csv("crowd_labels-improved.csv")
    # Remove the crowdlabels with content attributes
    df = df.loc[df["attribute-type"]!="content"]
    # Refine to positive crowd labels
    df = df.loc[df["label"]=="positive"]
    # Convert to appropriate datatype
    ins,outs = convert_output(df)
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
def img_id_to_dir(imgid):
    strid = imgid if type(imgid)==str else str(imgid)
    # Get the folder name from the id
    folder = strid[-4:]
    # Navigate to the data directory
    ddir = os.getcwd();ddir = op.dirname(ddir);ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data");ddir = op.join(ddir,"BAM");ddir = op.join(ddir,"images-annotated")
    # Get the image folder and directory
    ddir = op.join(ddir,folder)
    ddir = op.join(ddir,strid+".jpg")
    return ddir

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

###DEBUG###
# Open test image
def debug_load_image():
    ddir = get_bammaster_dir()
    ddir = op.join(ddir,"data");ddir = op.join(ddir,"obj");ddir = op.join(ddir,"train")
    # Get random image folder
    fold = rand.choice(os.listdir(ddir))
    ddir = op.join(ddir,fold)
    # Get a random image
    im = rand.choice(os.listdir(ddir))
    ddir = op.join(ddir,im)
    return load_img(ddir)
### DEBUG END

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
    ddir = os.getcwd();ddir = os.listdir(ddir);ddir = os.listdir(ddir)
    ddir = os.path.join(ddir,"Data");ddir = os.path.join(ddir,"BAM")
    return ddir

# Load the pre-trained BAM model
def load_bam_model(choice="obj"):
    ddir = get_bammaster_dir()
    ddir = op.join(ddir,"models")
    ddir = op.join(ddir,choice)
    return tf.saved_model.load(ddir)

# Get the directory of the bam-master folder
def get_bammaster_dir():
    ddir = os.getcwd()
    ddir = op.dirname(ddir)
    ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data")
    return op.join(ddir,"bam-master")

if(__name__=="__main__"): main()