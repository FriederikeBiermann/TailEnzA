
import Bio
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
import re
import numpy as np
from feature_generation import *
fastas_aligned_before=True
include_charge_features=False
#fill in filenames here!
foldernameoutput="Classifiers/"
foldername_training_sets="Training_data/Toy_dataset/toy_"
filename_permutations="permutations.txt"

enzymes=["p450","YCAO"]
BGC_types=["ripp","nrp","pk"]
model_proteins_for_alignment={"p450":"MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
                              "YCAO": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY"  }
start=0
end=400
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]], "YCAO": [["f1",start,92],["f2",93,192],["f3",193,275],["core",276,395],["end",396,end]]}
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"], "YCAO":["f1","f2","f3","core","end"]}



with open(filename_permutations, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]
def create_filenames(enzymes, BGC_types,foldername_training_sets):
    filenames=[]
    if fastas_aligned_before==False:
        for BGC in BGC_types:
            for enzyme in enzymes:
                filenames+=[foldername_training_sets+BGC+"_"+enzyme+".fasta"]
    if fastas_aligned_before==True:
        for BGC in BGC_types:
            for enzyme in enzymes:
                filenames+=[foldername_training_sets+BGC+"_"+enzyme+" alignment.fasta"]
    return filenames
for enzyme in enzymes:
    complete_feature_matrix=pd.DataFrame()
    path_complete_feature_matrix=foldernameoutput+enzyme+"_complete_feature_matrix.csv"
    filenames_training_set=create_filenames([enzyme],BGC_types,foldername_training_sets)
    print (filenames_training_set)
    for BGC_index, dataset in enumerate(filenames_training_set):
        
        if fastas_aligned_before==True:
            alignment = AlignIO.read(open(dataset), "fasta")
            fragment_matrix=fragment_alignment(alignment,splitting_lists[enzyme], fastas_aligned_before)


        if fastas_aligned_before==False:
            fragment_matrix=pd.DataFrame()
            seq_record_ids=[]
            for seq_record in SeqIO.parse(dataset, "fasta"):
           
                 fewgaps = lambda x, y: -20 - y
                 specificgaps = lambda x, y: (-2 - y)
                 alignment = pairwise2.align.globalmc(model_proteins_for_alignment[enzyme], seq_record.seq, 1, -1, fewgaps, specificgaps)
                 fragment_matrix_for_record=fragment_alignment (alignment[0],splitting_lists[enzyme],fastas_aligned_before)
                 fragment_matrix=fragment_matrix.append(fragment_matrix_for_record, ignore_index = True)
                 seq_record_ids=seq_record_ids+[seq_record.id]
    
        feature_matrix=featurize(fragment_matrix, permutations, fragments[enzyme], include_charge_features)
        feature_matrix["target"] = BGC_types[BGC_index]
   
        complete_feature_matrix=complete_feature_matrix.append(feature_matrix, ignore_index = True)
    print (complete_feature_matrix)

    complete_feature_matrix.to_csv(path_complete_feature_matrix, index=False)     
