#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:40:54 2022

@author: joe
"""

from functions_main import *

# Class used for predictor models

class Predictor:
  def __init__(self,model_dir):
    self.model_dir = model_dir
    self.model = tf.keras.models.load_model(model_dir)
    p_start = model_dir.rfind("/")+1
    self.model_name = model_dir[p_start:]
    return