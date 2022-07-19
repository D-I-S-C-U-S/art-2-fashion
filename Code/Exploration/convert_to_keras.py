#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:33:53 2022

@author: joe
"""

import tensorflow as tf
from os import path as op
import os

def get_data_dir():
    ddir = os.getcwd()
    ddir = op.dirname(ddir)
    ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data")
    return op.join(ddir,"bam-master")

def load_model(choice="obj"):
    ddir = get_data_dir()
    ddir = op.join(ddir,"models")
    ddir = op.join(ddir,choice)
    return tf.saved_model.load(ddir)

# Get an imagenet-trained ResNet50 model
def get_rn50():
    return tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224,224,3),
        pooling=None,
        classes=1000)

loaded = load_model()

rn = get_rn50()
print(rn.summary())
