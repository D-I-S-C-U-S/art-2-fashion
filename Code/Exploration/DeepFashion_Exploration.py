# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:57:16 2022

@author: joeda
"""

import pandas as pd
import os
from os import path as op

def main():
    get_data()
    return

# Navigate to the main data directory (will need changing if directory structure relative to data is changed)
def get_datadir():
    ddir = os.getcwd()
    ddir = op.dirname(ddir)
    ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data")
    ddir = op.join(ddir,"DeepFashion")
    return ddir

# Get a certain data file
def get_data(datafolder_s="CaTBench",folder_s="main",subfolder_s="anno-coarse",file_s="attr_img",):
    datadict = {"CaTBench":"Category and Attribute Prediction Benchmark"}
    datafolder = datadict[datafolder_s]
    folder = {"main":datadict[datafolder_s]}[folder_s]
    subfolder = {"anno-coarse":"Anno_coarse","anno-fine":"Anno_fine"}[subfolder_s]
    file = {"attr_img":"list_attr_img.txt"}[file_s]
    ddir = get_datadir()
    fdir = op.join(ddir,datafolder);fdir = op.join(fdir,folder);fdir = op.join(fdir,subfolder);fdir= op.join(fdir,file)
    with open(fdir,"r") as f:
        data = f.read()
    data = data.split("\n")
    DEBUG = 0
    datalist,maxlen = [],0
    for line in data:
        if(DEBUG>1):
            stuff = line.split(" ")
            # Remove blank lines from stuff
            i = 0
            while(i<len(stuff)):
                if(stuff[i]==""):
                    stuff.pop(i)
                else:
                    i += 1
            datalist.append(stuff.copy())
            maxlen = max(maxlen,len(stuff)-1)
        DEBUG += 1
        if(DEBUG>1000): break
    data = pd.DataFrame(datalist,columns=["Image Name"] + [f"Attribute label {i}" for i in range(1,maxlen+1)])
    "Category and Attribute Prediction Benchmark"
    "Consumer-to-shop Clothes Retrieval Benchmark"
    "Fashion Landmark Detection Benchmark"
    "Fashion Synthesis Benchmark"
    "In-shop Clothes Retrieval Benchmark"

if(__name__=="__main__"): main()