#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 08:26:52 2022

@author: friederike
"""


import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

def merge_two_dicts(x, y):
    #input-> 2 dictionaries output->  merged dictionary
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def calculate_charge(sequence):
    # uses aa sequence as input and calculates the approximate charge of it
    AACharge = {"C":-.045,"D":-.999,  "E":-.998,"H":.091,"K":1,"R":1,"Y":-.001}
    charge = -0.002
    seqstr=str(sequence)
    seqlist=list(seqstr)
    for aa in seqlist:
        if aa in AACharge:
            charge += AACharge[aa]
    return charge

def easysequence (sequence):
    #creates a string out of the sequence file, that only states if AA is acidic (a), basic (b), polar (p), neutral/unpolar (n),aromatic (r),Cystein (s) or a Prolin (t)
    seqstr=str(sequence)
    seqlist=list(seqstr)
    easylist=[]
    for i in seqlist:
        if i == 'E' or i== 'D':
            easylist=easylist+['a']
        if i == 'K' or i=='R' or i=='H':
            easylist=easylist+['b']
        if i == 'S' or i=='T' or i=='N' or i=='Q':
            easylist=easylist+['p']
        if i == 'F' or i=='Y' or i=='W':
            easylist=easylist+['r']
        if i == 'C':
            easylist=easylist+['s']
        if i == 'P':
            easylist=easylist+['t']
        if i == 'G' or i=='A' or i=='V' or i=='L' or i=='I' or i=='M':
            easylist=easylist+['n']

    seperator=''
    easysequence=seperator.join(easylist)
    return easysequence
def indexing_reference(record):
    # index the reference sequence without ignoring gaps
    list_reference=list(str(record.seq))

    index_aa=0
    index_mapping=[]
    for index,AA in enumerate(list_reference):
        if AA !="-":
            index_aa+=1
            index_mapping.append([index_aa,index])

    return (index_mapping)

def convert_splitting_list(splitting_list,index_reference):
    #-> convert the canonic splitting list to also reflect eventual gaps in the reference sequence
    converted_splitting_list=[]
    for fragment,[begin, end] in splitting_list.items():
        converted_splitting_list.append([fragment,index_reference[begin-1][1],index_reference[end-1][1]])
    return converted_splitting_list

def split_alignment(alignment, fragment, fastas_aligned_before):
    """
    Splits the aligned sequences at the positions determined by the splitting list.

    Parameters:
    - alignment: A list of SeqRecord objects or a single SeqRecord object.
    - fragment: A list with format [position_name, start, end] detailing where to split.
    - fastas_aligned_before: A boolean indicating if multiple sequences are provided.

    Returns:
    - A dictionary with record IDs as keys and the corresponding subsequences as values.
    """
    
    start = fragment[1]
    end = fragment[2]
    if not fastas_aligned_before:
        alignment = [alignment]
    
    seqRecord_dict_per_fragment = {}
    
    if fragment[0] == "begin":
        start = 1
        
    if fragment[0] != "end":
        for record in alignment:
            if record.id != "Reference":
                subsequence = str(record.seq)[start-1:end-1].replace('-', '')
                seqRecord_dict_per_fragment[record.id] = subsequence
    else:
        for record in alignment:
            if record.id != "Reference":
                subsequence = str(record.seq)[start-1:].replace('-', '')
                seqRecord_dict_per_fragment[record.id] = subsequence

    return seqRecord_dict_per_fragment

def fragment_alignment(alignment, splitting_list, fastas_aligned_before):
    """
    Creates a DataFrame matrix from the splitted alignment.

    Parameters:
    - alignment: A list of SeqRecord objects.
    - splitting_list: A list that determines where sequences should be split.
    - fastas_aligned_before: A boolean indicating if multiple sequences are provided.

    Returns:
    - A DataFrame with record IDs as index and fragments as columns.
    """

    fragment_matrix = pd.DataFrame()
    
    if not fastas_aligned_before:
        seqa = alignment[0]
        seqb = alignment[1]
        index_reference = indexing_reference(SeqRecord(Seq(seqa), id=seqa))
        converted_splitting_list = convert_splitting_list(splitting_list, index_reference)
        
        for fragment in converted_splitting_list:
            name_fragment = fragment[0]
            seqRecord_dict_per_fragment = split_alignment(SeqRecord(Seq(seqb), id=seqb), fragment, fastas_aligned_before)
            fragment_matrix[name_fragment] = pd.Series(seqRecord_dict_per_fragment)
    else:
        for record in alignment:
            if record.id == "Reference":
                index_reference = indexing_reference(record)
                converted_splitting_list = convert_splitting_list(splitting_list, index_reference)
                
                for fragment in converted_splitting_list:
                    name_fragment = fragment[0]
                    seqRecord_dict_per_fragment = split_alignment(alignment, fragment, fastas_aligned_before)
                    fragment_matrix[name_fragment] = pd.Series(seqRecord_dict_per_fragment)
                break

    return fragment_matrix

def featurize(fragment_matrix, permutations, fragments, include_charge_features):
    """
    Creates a feature matrix from the fragment matrix by counting motifs in each fragment.

    Parameters:
    - fragment_matrix: DataFrame of sequences with fragments as columns.
    - permutations: List of motifs to count in each fragment.
    - fragments: List of fragment names.
    - include_charge_features: A boolean to determine whether to include charge features.

    Returns:
    - A DataFrame representing the feature matrix.
    """
    if fragment_matrix.empty:
        return pd.DataFrame()
    feature_matrix = pd.DataFrame()
    new_rows = []
    
    for index, row in fragment_matrix.iterrows():
        new_row = {}
        for fragment in fragments:
            sequence_fragment = row[fragment]
            easysequence_fragment = easysequence(sequence_fragment)
            
            for motif in permutations:
                name_column = motif + fragment
                new_row[name_column] = easysequence_fragment.count(motif)
            if include_charge_features:
                new_row = append_charge_features(new_row, fragment, easysequence_fragment, sequence_fragment)
        new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows) # Assuming new_rows has the same columns as feature_matrix
    feature_matrix = pd.concat([feature_matrix, new_rows_df], ignore_index=True)

    if include_charge_features:
        feature_matrix = sum_charge_features(feature_matrix, fragments)

    return feature_matrix

def append_charge_features(new_row,fragment,easysequence_fragment,sequence_fragment):
    #append features indicating the charge to the feature matrix
    acidic=fragment+"acidic"
    new_row =merge_two_dicts(new_row,{acidic:(easysequence_fragment.count("a")/(len(easysequence_fragment)+1))})
    acidic_absolute=fragment+"acidic absolute"
    new_row =merge_two_dicts(new_row,{acidic_absolute:(easysequence_fragment.count("a"))})
    charge_name=fragment+"charge"
    new_row =merge_two_dicts(new_row,{charge_name:(calculate_charge(sequence_fragment))})
    basic=fragment+"basic"
    basic_absolute=fragment+"basic absolute"
    new_row =merge_two_dicts(new_row,{basic:(easysequence_fragment.count("b")/(len(easysequence_fragment)+1))})
    new_row =merge_two_dicts(new_row,{basic_absolute:(easysequence_fragment.count("b"))})
    return new_row

def sum_charge_features(feature_matrix, fragments):
    #sum up charge features to obtain the charge of the whole protein
    chargerows=[]
    acidicrows=[]
    basicrows=[]
    absacidicrows=[]
    absbasicrows=[]
    for fragment in fragments:
        chargerows.append(str(fragment)+"charge")
        acidicrows.append(str(fragment)+"acidic")
        basicrows.append(str(fragment)+"basic")
        absacidicrows.append(str(fragment)+"acidic absolute")
        absbasicrows.append(str(fragment)+"basic absolute")
    feature_matrix['complete charge']=feature_matrix[chargerows].sum(axis=1)
    feature_matrix['mean acidic']=feature_matrix[acidicrows].mean(axis=1)
    feature_matrix['mean basic']=feature_matrix[basicrows].mean(axis=1)
    feature_matrix['absolute acidic']=feature_matrix[absacidicrows].sum(axis=1)
    feature_matrix['absolute basic']=feature_matrix[absbasicrows].sum(axis=1)
    return feature_matrix
