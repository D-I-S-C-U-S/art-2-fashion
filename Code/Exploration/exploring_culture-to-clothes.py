# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:43:52 2022

@author: joeda
"""

import pickle
import os
from os import path as op
import pandas as pd

def main():
    # Get metadata
    meta,yearframe = meta_import()
    print(meta.columns)
    print(f"Total of {len(meta)} users, of which:\n\
          {len(meta['users'].unique())} are unique users, \n\
          {len(meta['ids'].unique())} are unique ids, \n\
          {len(meta['tags'].unique())} are unique tags, \n\
          {len(meta['titles'].unique())} are unique titles")
    
    

def get_ddir():
    ddir = os.getcwd()
    ddir = op.dirname(ddir)
    ddir = op.dirname(ddir)
    ddir = op.join(ddir,"Data")
    ddir = op.join(ddir,"culture-2-clothing_data")
    return ddir

def nyt_import(datadir=get_ddir(),years=[1900]):
    ddir = op.join(datadir,"nyt_corpus")
    stringyears = [str(year) for year in years]
    out = []
    for year in stringyears:
        docname = f"._doc_date_bows_year{year}.pkl"
        with open(op.join(ddir,docname),"rb") as f:
            out.append(pickle.load(f))
    if(len(out)==1):
        return out[0]
    else:
        return out

def meta_import(datadir=get_ddir()):
    ddir = op.join(datadir,"flickr_vintage")
    metadir,yeardir = op.join(ddir,"flickr_metadata.csv"),op.join(ddir,"person_year_dict.json")
    meta = pd.read_csv(metadir,sep="\t",index_col=0)
    yeardict = pd.read_json(yeardir,typ="series")
    yearframe = yeardict.to_frame()
    yearframe["Timestamp"] = yearframe.index
    yearframe["Year"] = [int(yearframe[0][i]) for i in range(len(yearframe))]
    t = list(yearframe["Timestamp"])
    y = list(yearframe["Year"])
    del yearframe
    yearframe = pd.DataFrame(data=zip(t,y),columns=["Timestamp","Year"])
    meta.drop_duplicates();yearframe.drop_duplicates()
    return meta,yearframe
    
    
if(__name__=="__main__"): main()