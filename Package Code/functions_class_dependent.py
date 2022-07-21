#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:43:57 2022

@author: joe
"""

from functions_main import *
from class_predictor import *

def predict_features(imgdirs,
                     emote_predictor = False,
                     media_predictor = False,
                     emote_save_dir = False,
                     media_save_dir = False):
  # Get predictors
  if(type(emote_predictor)==bool):
      emote_predictor = Predictor("prototype-emotion-1")
  if(type(media_predictor)==bool):
      media_predictor = Predictor("prototype-media-1")
  # Convert the image directories to tensors
  tens = dirs_to_tensors(imgdirs)
  # Predict this list of tensors
  emote_pred = predict_emotion(tens,emote_predictor)
  media_pred = predict_media(tens,media_predictor)
  # Delete the initial tensors
  del tens
  # Save the dataframe if appropriate
  if(type(emote_save_dir)==str):
    with open(emote_save_dir,"w") as f:
      emote_pred.to_csv(f,index=False,line_terminator="\n")
  if(type(media_save_dir)==str):
    with open(media_save_dir,"w") as f:
      media_pred.to_csv(f,index=False,line_terminator="\n")
  # Convert the predictions to single 11-column tensors
  tens = []
  for ind in emote_pred.index:
    tens.append(tf.convert_to_tensor(list(emote_pred.iloc[ind])+list(media_pred.iloc[ind])))
  tens = tf.convert_to_tensor(tens)
  return tens

# Get emotion and media predictions for a list of image dirs
def predict_styles(imgdirs=False,
                    emote_predictor = False,
                    media_predictor = False,
                    style_predictor = False,
                    emote_save_dir = False,
                    media_save_dir = False,
                    output_styles = True,
                    style_save_dir = False,
                    DEBUG = False):
  # Get predictors
  if(type(emote_predictor)==bool):
      emote_predictor = Predictor("prototype-emotion-1")
  if(type(media_predictor)==bool):
      media_predictor = Predictor("prototype-media-1")
  if(type(style_predictor)==bool):
      style_predictor = Predictor("widths = 1024,2048,128;activations = relu,relu,relu;lrate = 0.0001;epochs = 50")
  if(type(imgdirs)==bool):
    imgdirs = [os.path.join("/content/drive/MyDrive/Data-Science/Art-to-fashion/Data/HipsterWars/images",imgname) for imgname in os.listdir("/content/drive/MyDrive/Data-Science/Art-to-fashion/Data/HipsterWars/images")]
  # If debug is true, shorten the imgdirs list
  if(DEBUG==True):
    imgdirs = imgdirs[:10]
  # Get the feature predictions for the images
  feature_tens = predict_features(imgdirs,emote_predictor,media_predictor,emote_save_dir,media_save_dir)
  # Predict the styles for these images
  style_tens = style_predictor.model.predict(feature_tens)
  return style_tens

def train_style_predictor(imgdirs=False,
                          emote_predictor = False,
                          media_predictor = False,
                          styledata = False,
                          nn_architecture = [[20],["softmax"]],
                          epochs = 50,
                          lrate = 1e-4,
                          DEBUG = False,
                          save_desired = False):
  # Get predictors
  if(type(emote_predictor)==bool):
      emote_predictor = Predictor("prototype-emotion-1")
  if(type(media_predictor)==bool):
      media_predictor = Predictor("prototype-media-1")
  # Get image directories
  if(type(imgdirs)==bool):
      [os.path.join("/content/drive/MyDrive/Data-Science/Art-to-fashion/Data/HipsterWars/images",imgname) for imgname in os.listdir("/content/drive/MyDrive/Data-Science/Art-to-fashion/Data/HipsterWars/images")]
  # If debug, reduce the number of images
  if(DEBUG==True):
    imgdirs = imgdirs[:10]
  # If there is no predicted style file, get the predictions using the predictors
  if(type(styledata)==bool):
    feature_tens = predict_features(imgdirs,emote_predictor,media_predictor)
  # If a file for the style predictions exists, read it
  else:
    df = pd.read_csv(open(styledata,"r"))
    feature_tens = tf.convert_to_tensor(df["output"])
  # Get the correct style labels from the list of image directories
  imgids = [img_dir_to_id(imgdir) for imgdir in imgdirs]
  styledict = {"Hipster":0,"Goth":1,"Preppy":2,"Pinup":3,"Bohemian":4}
  labeltensorbase = [0 for i in range(len(styledict))]
  styleframe = pd.read_csv(open("hipster_to_csv_test.csv","r"))
  styleframe = styleframe[styleframe["ID"].isin(imgids)]
  outtens = []
  for imid in imgids:
    rframe = styleframe.loc[styleframe["ID"] == imid]
    label = list(rframe["Label"])[0]
    labelten = dc(labeltensorbase)
    labelten[styledict[label]] = 1
    outtens.append(tf.convert_to_tensor(labelten,dtype=float))
  outtens = tf.convert_to_tensor(outtens)
  # Create a neural network with the intended architecture - takes the form [[width of layer 1, width of layer 2,..., width of final layer],[activation of layer 1,.., activation of final layer]]
  model_inputs = tf.keras.Input(shape=(11,))
  model_hidden = [model_inputs]
  str_to_act = {"softmax":tf.nn.softmax,"relu":tf.nn.relu}
  for i in range(len(nn_architecture[0])):
    try:
      model_hidden.append(tf.keras.layers.Dense(nn_architecture[0][i],nn_architecture[1][i])(model_hidden[-1]))
    except:
      model_hidden.append(tf.keras.layers.Dense(nn_architecture[0][i],tf.nn.softmax)(model_hidden[-1]))
  model_outputs = tf.keras.layers.Dense(5,activation=tf.nn.softmax)(model_hidden[-1])
  # Train the model
  totrain = tf.keras.Model(inputs=model_inputs,outputs=model_outputs)
  totrain.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=lrate),loss=tf.keras.losses.BinaryCrossentropy(),
             metrics=[tf.keras.metrics.BinaryAccuracy(),
                      tf.keras.metrics.FalseNegatives()])
  history = totrain.fit(x=feature_tens,y=outtens,epochs=epochs,validation_split=0.2)
  plot_history(history,f"Performance for style training")
  # Save the model
  if(save_desired==True):
    filename = f"widths = {','.join([str(width) for width in nn_architecture[0]])};activations = {','.join(nn_architecture[1])};lrate = {lrate};epochs = {epochs}"
    totrain.save(filename)
  return totrain

### DEBUG - ONLY NEEDS TO BE RUN ONCE ###
def predict_clothes(emote_predictor = False,
                    media_predictor = False,
                    style_predictor = False,
                    imgdirs = False,
                    DEBUG = False):
  # Get predictors
  if(type(emote_predictor)==bool):
      emote_predictor = Predictor("prototype-emotion-1")
  if(type(media_predictor)==bool):
      media_predictor = Predictor("prototype-media-1")
  if(type(style_predictor)==bool):
      style_predictor = Predictor("widths = 1024,2048,128;activations = relu,relu,relu;lrate = 0.0001;epochs = 50")
  # Isolate the individual model names and make the filename
  emote_name,media_name,style_name = emote_predictor.model_dir,media_predictor.model_dir,style_predictor.model_dir
  e_start,m_start,s_start = emote_name.rfind("/")+1,media_name.rfind("/")+1,style_name.rfind("/")+1
  emote_name,media_name,style_name = emote_name[e_start:],media_name[m_start:],style_name[s_start:]
  filename = f"emote_predictor,{emote_name};media_predictor,{media_name};style_predictor,{style_name}.csv"
  filedir = filename
  # Get the image directories for all clothes present and get their features
  if(type(imgdirs)==bool):
      imgdirs = [os.path.join("/content/drive/MyDrive/Data-Science/Art-to-fashion/Data/HipsterWars/images",imgname) for imgname in os.listdir("/content/drive/MyDrive/Data-Science/Art-to-fashion/Data/HipsterWars/images")]
  if(DEBUG==True):
    imgdirs = imgdirs[:10]
  imgids = [img_dir_to_id(imgdir) for imgdir in imgdirs]
  feature_tens = predict_features(imgdirs,emote_predictor,media_predictor)
  style_tens = style_predictor.model.predict(feature_tens)
  style_tens = list(style_tens)
  feature_frame = pd.DataFrame(data=list(zip(imgdirs,imgids,style_tens)),columns=["directory","ID","styles"])
  with open(filedir,"w") as f:
    feature_frame.to_csv(f,index=False,line_terminator="\n")
  return