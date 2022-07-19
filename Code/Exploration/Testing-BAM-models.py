#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:47:07 2022

@author: joe

Testing/experimenting with the models taken from the BAM paper
"""

import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot_keras_model
import numpy as np
import os
from os import path as op
import random as rand
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import time as t

# Taken from BAM attributions.py; same as in https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_preprocessing.py
RESNET_SHAPE = (224, 224)
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
"""
    img = PIL.Image.open(tf.gfile.Open(f, 'rb'))
    img = img.convert('RGB').resize(RESNET_SHAPE, PIL.Image.BILINEAR)
"""

# Load image, taken from BAM attributions.py
def load_img(f):
  img = Image.open(tf.io.gfile.GFile(f, 'rb'))
  img = img.convert('RGB').resize(RESNET_SHAPE, Image.BILINEAR)
  channel_means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
  img_arr = np.array(img, dtype=np.float32) - channel_means.astype(np.float32)
  return img_arr

def main():
    test = load_model()
    #plot_keras_model(test,"BAM_model.png",show_shapes = True,show_layer_names = True,show_layer_activations = True)
    print(test.signatures)
    predf = test.signatures["predict"]
    print(test.summary())
    #print(predf.metrics)
    return

# Produce predictions for all training images and save as csv files
# Folders is used to determine training sources
def predict_trainings(folders=["obj"],desired_model="obj"):
    # Load the desired model
    test_model = load_model(desired_model)
    # Begin by checking that the correct directories exist, and making them if not
    ddir = get_data_dir()
    if("bam-preds" not in os.listdir(ddir)):
        os.mkdir(op.join(ddir,"bam-preds"))
    targdir = op.join(ddir,"bam-preds")
    maindir = op.join(ddir,"data")
    # Go through folders and predict results
    # Results will take the form [filename,model outputs 1-10]
    results = []
    for folder in folders:
        ddir = op.join(maindir,folder)
        # Ensure there is a training folder
        if("train" not in os.listdir(ddir)): continue
        ddir = op.join(ddir,"train")
        for obj_class in os.listdir(ddir):
            print(f"Predicting {obj_class} images...")
            temptime,t2time,t2record = t.time(),t.time(),0
            classdir = op.join(ddir,obj_class)
            for obj in os.listdir(classdir):
                img = load_img(op.join(classdir,obj))
                prediction = predict_base(img,model=test_model).numpy().flatten()
                entry = [obj] + list(prediction)
                results.append(entry.copy())
                t2record += 1
                if(t.time()-t2time>300):
                    t2time = t.time()
                    pdone = round(np.divide(t2record,len(os.listdir(classdir)))*100,1)
                    tdone = round(np.divide(t.time()-temptime,60),0)
                    print(f"{pdone}% of current predictions complete after {tdone} minutes")
            if(f"{obj_class}-preds" not in os.listdir(targdir)):
                os.mkdir(op.join(targdir,f"{obj_class}-preds"))
            filedir = op.join(targdir,f"{obj_class}-preds")
            filename = f"{desired_model}-trained_predictions.csv"
            filedir = op.join(filedir,filename)
            results = pd.DataFrame(results,columns=["filename"]+[f"N{i} Output" for i in range(len(entry)-1)])
            results.to_csv(open(filedir,"w"),index=False,line_terminator="\n")
            print(f"Saved {filedir} after {round(np.divide(t.time()-temptime,60),1)} minutes of processing")

# Predict a test image from one of the 10 training groups
def DEBUG_pred(fdir="obj",group="bird"):
    # Get to the correct directory.
    ddir = get_data_dir()
    ddir = op.join(ddir,"data")
    if(fdir in os.listdir(ddir)):
        ddir = op.join(ddir,fdir)
    else:
        ddir = op.join(ddir,"obj")
        print("Unrecognised file directory; defaulting to obj")
    ddir = op.join(ddir,"train")
    if(group in os.listdir(ddir)):
        ddir = op.join(ddir,group)
    else:
        ddir = op.join(ddir,"bird")
        print("Unrecognised group; defaulting to bird")
    # Select a random image and predict it
    choice = rand.choice(os.listdir(ddir))
    img = load_img(op.join(ddir,choice))
    return predict_base(img)

# Predict a single test image from each of the 10 training groups
def DEBUG_predall(fdir="obj"):
    # Navigate to the appropriate folder to find all training groups
    ddir = get_data_dir();ddir = op.join(ddir,"data")
    if(fdir in os.listdir(ddir)):
        ddir = op.join(ddir,fdir)
    else:
        ddir = op.join(ddir,"obj")
        fdir = "obj"
        print("Unrecognised file directory; defaulting to obj")
    ddir = op.join(ddir,"train")
    # Gather results
    results = {}
    for objname in os.listdir(ddir):
        results[objname] = DEBUG_pred(fdir,objname)
    return results

# Display the predictions for one of each training image
def DEBUG_predall_display(fdir="obj",mode="single"):
    # Get results for a single random prediction
    if(mode=="single"):
        results = DEBUG_predall(fdir)
        for key in results: results[key] = results[key].numpy().flatten()
    # Get results for a number of random predictions averaged
    else:
        raw_results,results = [],{}
        if(mode in {"some","many"}): mode = {"some":10,"many":100}[mode]
        for i in range(mode):
            result = DEBUG_predall(fdir)
            for key in result: result[key] = result[key].numpy().flatten()
            raw_results.append(result)
        for key in raw_results[0]:
            temp = [raw_results[i][key] for i in range(len(raw_results))]
            new = np.zeros(len(temp[0]),dtype=float)
            for i in range(len(new)):
                newnum = []
                for j in range(len(temp)):
                    newnum.append(temp[j][i])
                new[i] = np.mean(newnum)
            results[key] = new.copy()
    # Plot the results
    fig,ax = plt.subplots()
    for key in results:
        ax.scatter(range(results[key].shape[0]),results[key],label=key)
    ax.set_title("Extracted probability scores for randomly chosen training images")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Neuron")
    plt.legend(loc="center right")
    plt.show()
    return
    
    



# Get the directory of the bam-master folder
def get_data_dir():
    ddir = os.getcwd()
    ddir = op.dirname(ddir)
    ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data")
    return op.join(ddir,"bam-master")

# Get directory to models
def load_model(choice="obj"):
    ddir = get_data_dir()
    ddir = op.join(ddir,"models")
    ddir = op.join(ddir,choice)
    return tf.saved_model.load(ddir)

# Get directory to images
def load_images(choice="coco"):
    ddir = get_data_dir()
    ddir = op.join(ddir,"data")
    return op.join(ddir,choice)

# Predict coco data for exploration
def predict_coco_debug(model=load_model(),mode="some"):
    ddir = load_images()
    ddir = op.join(ddir,"images");ddir = op.join(ddir,"train2017")
    files = os.listdir(ddir)
    if(mode=="one" or mode=="some"):
        for _ in range({"one":1,"some":10}[mode]):
            choice = op.join(ddir,rand.choice(files))
            img = load_img(choice)
            print(predict_base(img,model))
    return

# Predict an image using one of the pre-existing models
def predict_base(img,model=load_model()):
    imgtens = tf.convert_to_tensor([img],dtype=float)
    return model.signatures['predict'](imgtens)["probabilities"]
        
    

if(__name__=="__main__"): main()