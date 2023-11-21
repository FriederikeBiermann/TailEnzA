#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 08:42:41 2022
' formats genbank files so only features completely within sequence are inlcuded
@author: friederike
"""


import os
from Bio import SeqIO

foldername_input= "/home/friederike/Documents/Coding/TailEnzA_main/TailEnzA/Classifiers/Training_data/Dataset/Terpenes_genbank_files_antismash_DB/"
list_filenames=os.listdir(foldername_input)
foldername_output= "/home/friederike/Documents/Coding/TailEnzA_main/TailEnzA/Classifiers/Training_data/Dataset/Terpenes_genbank_files_antismash_DB_corrected/"
def remove_features_out_of_bounds(genbank):
    #print(genbank.features[0])
    begin_sequence=genbank.features[0].location.start
    end_sequence=genbank.features[0].location.end
    trimmed_features=[]
    for feature in genbank.features:
        #print(feature)
        if feature.location.start>=begin_sequence and feature.location.end>=begin_sequence and feature.location.start<=end_sequence and feature.location.end<=end_sequence:
            trimmed_features+=[feature]
    #print(trimmed_features)
    genbank.features=trimmed_features
    return genbank
for filename in list_filenames:
    #print (filename)
    try:
        for genbank in SeqIO.parse(foldername_input+filename, "genbank"):
            genbank_new=remove_features_out_of_bounds(genbank)
        SeqIO.write(genbank_new, foldername_output+filename, "genbank")
    except: print("error",filename)
