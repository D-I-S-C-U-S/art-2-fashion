#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:47:07 2022

@author: joe

Testing/experimenting with the models taken from the BAM paper
"""

import tensorflow as tf
import numpy as np
import os
from os import path as op
import random as rand
from PIL import Image

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
    print(load_model().signatures['predict'])
    predict_coco(mode="one")
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

# Predict coco data
### CURRENTLY PRODUCES ERROR: The first dimension of paddings must be the rank of inputs[4,2] [224,224,3] ###
def predict_coco(model=load_model(),mode="some"):
    ddir = load_images()
    ddir = op.join(ddir,"images");ddir = op.join(ddir,"train2017")
    files = os.listdir(ddir)
    if(mode=="one" or mode=="some"):
        for _ in range({"one":1,"some":10}[mode]):
            choice = op.join(ddir,rand.choice(files))
            img = load_img(choice)
            imgtens = tf.convert_to_tensor(img,dtype=float)
            model.signatures['predict'](imgtens)
    return
        
    

if(__name__=="__main__"): main()