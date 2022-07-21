#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:59:10 2022

@author: joe
Main module to simply predict a style
"""

from functions_main import *
from class_predictor import *
from functions_class_dependent import *

# Take an input image directory and find the most similar clothing results using both predicted and true outputs
def predict_image_style(imgdir,
                        emote_predictor = Predictor("prototype-emotion-1"),
                        media_predictor = Predictor("prototype-media-1"),
                        style_predictor = Predictor("widths = 1024,2048,128;activations = relu,relu,relu;lrate = 0.0001;epochs = 50"),
                        prediction_weightings = 0.5):
  # Predict the style for this image
  feature_tens = predict_features([imgdir],emote_predictor,media_predictor)
  style_tens = style_predictor.model.predict(feature_tens)
  ## Get the actual and predicted styles of known clothes
  styles_actual = get_hipsterdata()
  styles_predicted_dir = f"emote_predictor,{emote_predictor.model_name};media_predictor,{media_predictor.model_name};style_predictor,{style_predictor.model_name}.csv"
  styles_predicted = pd.read_csv(styles_predicted_dir)
  # Create a combined dataframe
  actual_outs,actual_ids,pred_outs,pred_ids = list(styles_actual["style"]),list(styles_actual["ID"]),list(styles_predicted["styles"]),list(styles_predicted["ID"])
  # Sort both by ID
  actual_outs = [x for _,x in sorted(zip(actual_ids,actual_outs))]
  actual_ids = sorted(actual_ids)
  pred_outs = [x for _,x in sorted(zip(pred_ids,pred_outs))]
  del pred_ids
  # Start scoring them all by similarity and constantly print the best one
  best_score,bsi = np.inf,-1
  for i in range(len(actual_outs)):
    # Convert the pred_outs[i] string to a list
    po = pred_outs[i]
    po = po.split(" ")
    for j in range(len(po)):
      po[j] = po[j].replace("[","")
      po[j] = po[j].replace("]","")
    j = 0
    while(j<len(po)):
      if(type(po[j])==float):
        continue
      if(len(po[j])==0):
        po.pop(j)
        continue
      try:
        po[j] = float(po[j])
        j += 1
      except:
        po.pop(i)
    res = compare_to_styles(style_tens[0],actual_outs[i],po,pred_weight = prediction_weightings)
    # If the new result is more similar than the current best, report this, and update the best score + best score index
    if(res<=best_score):
      print(f"Current best clothing found: ID {actual_ids[i]} with similarity of {res}")
      best_score = res
      bsi = i
  return actual_ids[bsi]

predict_image_style("test-image.jpg")