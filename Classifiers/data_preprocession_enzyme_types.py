
import Bio
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from feature_generation import *

include_charge_features=False
#fill in filenames here!
foldernameoutput="Classifiers/Toy_dataset/Enzyme_recognition/"
foldername_training_sets="Training_data/Toy_dataset/all"
filename_permutations="permutations.txt"

enzymes=["p450","YCAO","SAM"]




with open(filename_permutations, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]
def create_filenames(enzymes,foldername_training_sets):
    filenames=[]
    for enzyme in enzymes:
        filenames+=[foldername_training_sets+"_"+enzyme+".fasta"]
    return filenames

complete_feature_matrix=pd.DataFrame()
path_complete_feature_matrix=foldernameoutput+"_complete_feature_matrix_enzymes.csv"
filenames_training_set=create_filenames(enzymes,foldername_training_sets)
print (filenames_training_set)
for enzyme_index, dataset in enumerate(filenames_training_set):
        fragment_matrix=pd.DataFrame()
        
        for index, record in enumerate(SeqIO.parse(dataset, "fasta")):
            new_row= {"whole_enzyme":record.seq}
            fragment_matrix=fragment_matrix.append(new_row, ignore_index=True)

        feature_matrix=featurize(fragment_matrix, permutations, ["whole_enzyme"], include_charge_features)
        feature_matrix["target"] = enzymes[enzyme_index]
        complete_feature_matrix=complete_feature_matrix.append(feature_matrix, ignore_index = True)
print (complete_feature_matrix)
complete_feature_matrix.to_csv(path_complete_feature_matrix, index=False)     
