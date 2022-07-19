#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:24:33 2022

@author: joe
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model as plot_keras_model
import pydot
import graphviz
from PIL import Image

RESNET_SHAPE = (224,224,3)

def main():
    model = content_attribute_classifier()
    #plot_keras_model(model,"Keras_Resnet-50.png",show_shapes = True,show_layer_names = True,show_layer_activations = True)
    img = Image.open(tf.io.gfile.GFile("test_image.jpg", 'rb'))
    img = img.convert('RGB')
    img = img.resize(size=(RESNET_SHAPE[0],RESNET_SHAPE[1]), resample=Image.BILINEAR)
    img_arr = np.array(img,dtype=np.float32)
    img_ten = tf.convert_to_tensor([img_arr])
    print(model.predict(img_ten))
    return

# Get resnet-50
def content_attribute_classifier(inshape = RESNET_SHAPE):
    return tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=inshape,
    pooling=None,
    classes=1000)

# Get stylenet
def emotion_media_attirbute_classifier():
    # Style net code doesn't seem to be accessible
    # Created from fine-tuning GoogLeNet (Going Deeper with Convolutions by Szegedy et al.;
    # Tensorflow implementation available from https://github.com/conan7882/GoogLeNet-Inception)
    # on social media from behance (Collaborative Feature Learning from Social Media by Fang et al.)
    # (Behance available at https://www.behance.net/)
    return
    
if(__name__=="__main__"): main()