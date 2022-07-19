# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:20:06 2022

@author: joeda
"""

import pandas as pd
from scipy.io import loadmat
import os
from os import path as op
from copy import deepcopy as dc
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt

def main():
    #convert_to_csv()
    #test = read_csv()
    #plot_label_counts()
    show_corrs()
    return

# Plot the correlation between the three features, the accuracy for a neural and logistic regressor, and plot correct labels
def show_corrs():
    data = read_csv()
    data = data.drop(labels=["ID","hyperlink"],axis="columns")
    train,test = train_test_split(data,test_size=0.5,random_state=0)
    print(train.corr())
    x_train,y_train,x_test,y_test = \
        list(zip(train["Val 1"],train["Val 2"],train["Val 3"])),\
        list(train["Label"]),\
        list(zip(test["Val 1"],test["Val 2"],test["Val 3"])),\
        list(test["Label"])
    # Convert the y values to integers
    def make_label_numbers(groups):
        label_numbers,out = {},[]
        for group in groups:
            for i in range(len(group)):
                if(group[i] not in label_numbers):
                    label_numbers[group[i]] = len(label_numbers)
                group[i] = label_numbers[group[i]]
            out.append(group.copy())
        return out,label_numbers
    [y_train,y_test],label_numbers = make_label_numbers([y_train,y_test])
    number_labels = {}
    for label in label_numbers:
        number_labels[label_numbers[label]] = label
    nn = sklearn.neural_network.MLPRegressor(random_state=0,hidden_layer_sizes=(20,),max_iter=2500)
    nn.fit(x_train,y_train)
    print(f"Score for an SKLearn Neural Regressor predicting label from Values: {nn.score(x_train,y_train)}")
    clf = LogisticRegression(random_state = 0,max_iter=2500)
    clf.fit(x_train,y_train)
    print(f"Score for an SKLearn Logistic Regressor predicting label from Values: {clf.score(x_train,y_train)}")
   
    # Plot label counts and errors
    preds = clf.predict(x_train)
    ys = {}
    for label in label_numbers:
        ys[label] = [0,0,0]
    for i in range(len(x_train)):
        label = number_labels[y_train[i]]
        pred = number_labels[preds[i]]
        # Add for total frequency
        ys[label][0] += 1
        # Add correct predictions
        if(label==pred):
            ys[label][1] += 1
        # Add erroneous labels
        elif(label!=pred):
            ys[pred][2] += 1
            
    # Plot the data
    xs = list(ys.keys())
    counts = [ys[label][0] for label in list(ys.keys())]
    corrects = [ys[label][1] for label in list(ys.keys())]
    wrongs = [ys[label][2] for label in list(ys.keys())]
    
    x = np.arange(len(xs))
    width = 0.25
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, counts, width, label='Count')
    rects2 = ax.bar(x, corrects, width, label='Correct Predictions')
    rects3 = ax.bar(x + width, wrongs, width, label = "Incorrect Predictions")
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Actual and Correctly predicted labels for training set')
    ax.set_xticks(x,xs)
    ax.set_xlabel("Label")
    ax.legend(loc = "lower left")
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    
    fig.tight_layout()
    
    plt.show()
    return
    
# Plot the counts of different labels for the data
def plot_label_counts():
    data = read_csv()
    ax = data["Label"].value_counts().plot(kind="bar",title="Frequency of labels within hipsterwars")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    return

# Read the to_test csv file
def read_csv():
    # Get csv directory
    d = os.getcwd();d=op.dirname(d);d=op.dirname(d);d=op.join(d,"Data");d=op.join(d,"hipster_to_csv_test.csv")
    # Open as pandas dataframe
    with open(d,"r") as f:
        data = pd.read_csv(f)
    return data


# Convert a .mat file to a csv for easier loading
def convert_to_csv():
    # Get file directory
    ddir = os.getcwd()
    ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data")
    fdir = op.join(ddir,"hipsterwars_Jan_2014.mat")
    data = loadmat(fdir)
    out = []
    pchecki,pcheck = 0,0.0
    # Note: RGB data is lost due to data format combined with time necessary to process
    for entry in data["samples"]:
        new,rel = [],entry[0]
        new.append(rel[0][0][0])
        new.append(rel[1][0])
        new.append(rel[2][0][0])
        new.append(rel[3][0][0])
        new.append(rel[4][0][0])
        new.append(rel[5][0])
        pchecki += 1
        if(pchecki/len(data["samples"])>=pcheck):
            print(f"{round(pcheck*100,1)}% converted")
            pcheck += 0.1
        out.append(new.copy())
    out = pd.DataFrame(data=out,columns=["ID","Label","Val 1","Val 2","Val 3","hyperlink"])
    # Get to data directory
    d = os.getcwd();d=op.dirname(d);d=op.join(d,"Data");d=op.join(d,"hipster_to_csv_test.csv")
    with open(d,"w") as f:
        out.to_csv(f,index=False,line_terminator="\n")
    return

if(__name__=="__main__"): main()