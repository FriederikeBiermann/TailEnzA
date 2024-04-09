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
import torch
import esm

import numpy as np
import pandas as pd
from pathlib import Path

# Load the ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model = model.eval()
batch_converter = alphabet.get_batch_converter()

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
    print(record.id, record.seq)
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
  
    for fragment in splitting_list:
        print(fragment, converted_splitting_list, index_reference)
        converted_splitting_list.append([fragment[0],index_reference[fragment[1]-1][1],index_reference[fragment[2]-1][1]])
    return converted_splitting_list

def split_alignment(alignment, fragment, fastas_aligned_before):
    # split the aligned sequences at the positions determined by the splitting list
    start = fragment[1]
    end = fragment[2]
    if fastas_aligned_before == False:
        alignment = [alignment]
    seqRecord_list_per_fragment = []
    if fragment[0] == "begin":
        start = 1
    if fragment[0] != "end":
        for record in alignment:
            if record.id != "Reference":
                subsequence = str(record.seq)[start-1:end-1].replace('-', '')

                seqRecord_list_per_fragment.append(
                    [record.id, subsequence])
    else:
        for record in alignment:
            if record.id != "Reference":
                subsequence = str(record.seq)[start-1:].replace('-', '')
                seqRecord_list_per_fragment.append(
                                                        [record.id, subsequence])
    seqRecord_array_per_fragment = np.array(seqRecord_list_per_fragment)

    return seqRecord_array_per_fragment

def fragment_alignment(alignment,splitting_list, fastas_aligned_before):
    # create a matrix from the splitted alignment
    fragment_matrix=pd.DataFrame()
    if fastas_aligned_before==False:

        seqa=alignment[0]
        seqb=alignment[1]
        index_reference=indexing_reference(SeqRecord(Seq(seqa),id=seqa))

        converted_splitting_list=convert_splitting_list(splitting_list,index_reference)
        for fragment in converted_splitting_list:
                name_fragment=fragment[0]
                seqRecord_list_per_fragment=split_alignment(SeqRecord(Seq(seqb),id=seqb),fragment,fastas_aligned_before)

                fragment_matrix[name_fragment]=seqRecord_list_per_fragment[:,1]
                fragment_matrix.set_index(pd.Index(seqRecord_list_per_fragment[:,0]))
    else:
        for record in alignment:
            if record.id=="Reference":
                index_reference=indexing_reference(record)
                converted_splitting_list=convert_splitting_list(splitting_list,index_reference)
                for fragment in converted_splitting_list:
                    name_fragment=fragment[0]
                    seqRecord_list_per_fragment=split_alignment(alignment,fragment,fastas_aligned_before)
                    fragment_matrix[name_fragment]=seqRecord_list_per_fragment[:,1]
                fragment_matrix.set_index(pd.Index(seqRecord_list_per_fragment[:,0]))
                break

    return fragment_matrix




def generate_transformer_embeddings(sequence_labels, sequence_strs, batch_converter, model):
    """
    Generate transformer embeddings for a list of sequences.
    """
    batch_labels, batch_strs, batch_tokens = batch_converter([(sequence_labels, sequence_strs)])
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_embeddings = results['representations'][33]
    
    # Average embeddings across the sequence length for a single vector representation
    averaged_embeddings = token_embeddings.mean(dim=1)
    return averaged_embeddings

def featurize_fragments(fragment_matrix, batch_converter, model, include_charge_features=True):
    """
    Generate features for each fragment in the fragments dictionary.
    """
    for fragment_name in fragment_matrix.columns:
        sequence_strs = fragment_matrix[fragment_name].dropna().tolist()  # Ensure to drop any NaN values
        sequence_labels = [f"{fragment_name}_{i}" for i in range(len(sequence_strs))]
        
        # Generate embeddings
        _, _, batch_tokens = batch_converter([(sequence_labels, sequence_strs)])
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            token_embeddings = results["representations"][33]
        
        # Process embeddings
        for i, embedding in enumerate(token_embeddings):
            averaged_embedding = embedding.mean(dim=0).numpy().flatten()
            feature_row = {f"{fragment_name}_emb_{idx}": val for idx, val in enumerate(averaged_embedding)}
            
            if include_charge_features:
                sequence = sequence_strs[i]
                charge = calculate_charge(sequence)
                feature_row.update({
                    f"{fragment_name}_charge": charge,
                    # Additional charge features could be added here
                })
            
            feature_matrix = feature_matrix.append(feature_row, ignore_index=True)

    return feature_matrix