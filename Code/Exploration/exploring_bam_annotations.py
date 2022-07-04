#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:18:23 2022

@author: joe
"""

import os
from os import path as op
import pandas as pd

def main():
    annodir = anno_csv_dir()
    dfile = op.join(annodir,"crowd_labels.csv")
    test = pd.read_csv(dfile)
    #print(test.columns)
    #print(test["attribute"].unique())
    #print(test["label"].unique())
    ref = test.loc[test["label"]=="positive"]
    # Splite attribute into attribute type and attribute
    ref.insert(1,"attribute-type",[a.split("_")[0] for a in list(ref["attribute"])].copy())
    temp = [a.split("_")[1] for a in list(ref["attribute"])].copy()
    ref["attribute"] = temp.copy()
    del temp
    print(ref.columns)
    print(ref["attribute"].unique())
    print(ref["attribute-type"].unique())
    print(ref)
    temp = test["attribute"].unique()
    temp2 = {}
    for b in temp:
        _ = b.split("_")
        if(_[0]) not in temp2: temp2[_[0]] = []
        temp2[_[0]].append(_[1])
    for key in temp2:
        print(f"attributes of {key}: {temp2[key]}")

# Get the directory of the csv labels folder
def anno_csv_dir():
    ddir = os.getcwd();ddir = op.dirname(ddir);ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data");ddir = op.join(ddir,"BAM");ddir = op.join(ddir,"labels-csv")
    return ddir

if(__name__=="__main__"): main()