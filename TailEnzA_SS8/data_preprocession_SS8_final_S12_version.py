
import Bio
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from feature_generation_SS8 import *

# if fastas aligned before True-> use alignment made for instance in geneious utilizing MUSCLE align -> best for larger datasets
fastas_aligned_before=True
# if include_charge_features=True-> features describing general electrostatic information will be included
include_charge_features=True
#fill in filenames here!
foldernameoutput="Classifiers/SS8_antismash_dataset_matrices/gomin2_gemin1/"
foldername_training_sets="SS8_gomin2_gemin1_datasets/"#"/toy_" #an diesen string wird filenames angehschlossen
filename_permutations="permutations.txt"

enzymes=["p450","ycao","SAM","Methyl"]
#enzymes=["Methyl"]
BGC_types=["ripp","nrp","pk"]
## the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"p450":"MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
                              "ycao":"MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
                              "SAM":"MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL",
                              "Methyl":"MGSSHHHHHHSSGLVPRGSHMTTETTTATATAKIPAPATPYQEDIARYWNNEARPVNLRLGDVDGLYHHHYGIGPVDRAALGDPEHSEYEKKVIAELHRLESAQAEFLMDHLGQAGPDDTLVDAGCGRGGSMVMAHRRFGSRVEGVTLSAAQADFGNRRARELRIDDHVRSRVCNMLDTPFDKGAVTASWNNESTMYVDLHDLFSEHSRFLKVGGRYVTITGCWNPRYGQPSKWVSQINAHFECNIHSRREYLRAMADNRLVPHTIVDLTPDTLPYWELRATSSLVTGIEKAFIESYRDGSFQYVLIAADRV"}

start=0
end=350
# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]],
                 "ycao": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 "Methyl":[["begin",0,78],["SAM1",79,104],["SAM2",105,128],["SAM3",129,158],["SAM4",159,188],["SAM5",189,233],["end",234,255]]}
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"],
           "ycao":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "SAM":["begin","SAM","bridging","end"],
           "Methyl":["begin","SAM1","SAM2","SAM3","SAM4","SAM5","end"]
           }



#hier nichts verändern, liest das permutationsfile ein und überführt diese in eine Liste
#permutations= list of 4-aa motifs to use as features
with open(filename_permutations, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]

def create_filenames(enzymes, BGC_types,foldername_training_sets):

    #creates a list of all filenames of the input files -> change the exact expression if files named differently

    filenames_1=[]
    filenames_2=[]
    if fastas_aligned_before==True:
        for BGC in BGC_types:
            for enzyme in enzymes:
                filenames_2+=[foldername_training_sets+BGC+"_"+enzyme+".fasta"]
    if fastas_aligned_before==True:
        for BGC in BGC_types:
            for enzyme in enzymes:
                filenames_1+=[foldername_training_sets+BGC+"_"+enzyme+"_gomin2_gemin1.fas"]
    return filenames_1, filenames_2

def map_SS8_to_aa(header,fragment_indices, filename):

    with open(filename) as fasta_file:
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):
            if seq_record.id==header: #wenn die id des fasta files dem header in dem added_up_frame_reindexed entspricht
                aa_sequence= seq_record.seq #aa_sequence ist die sequenz, die zu der record id des headers gehört
                fragment_list=[] #erstellt neue leere Liste, in die geschnittene AA-Fragmente eingefügt werden sollen
                for index in fragment_indices: #für jeden Index in fragment
                    fragment=aa_sequence[:index]
                    aa_sequence=aa_sequence[index:]
                    fragment_list+=[str(fragment)]
                return fragment_list

for enzyme in enzymes:
    # the complete feature matrix is the input for all subsequent steps-> as we have different classifiers for each enzyme, we also need different training sets for each enzyme
    complete_feature_matrix=pd.DataFrame()
    path_complete_feature_matrix=foldernameoutput+enzyme+"_complete_feature_matrix.csv"
    filenames_training_set_1, filenames_training_set_2=create_filenames([enzyme],BGC_types,foldername_training_sets)
    print (filenames_training_set_1)
    #variable hier neu definiert
    #filenames_training_set=[foldername_training_sets+"Testfile_ripps_Methyl_SS8_aligned.fasta"]

    for BGC_index, dataset in enumerate(filenames_training_set_1):
        filename_AA=filenames_training_set_2[BGC_index]
        #fill in the feature matrix with data from files, the fragment matrix is a table with all sequences split into the different functional parts
        if fastas_aligned_before==True:
            alignment = AlignIO.read(open(dataset), "fasta")
            fragment_matrix=fragment_alignment(alignment,splitting_lists[enzyme]
            , fastas_aligned_before)



        if fastas_aligned_before==False:
            fragment_matrix=pd.DataFrame()
            seq_record_ids=[]
            for seq_record in SeqIO.parse(dataset, "fasta"):
                 # create custom alignment algorithm for the alignment against the reference sequence
                 fewgaps = lambda x, y: -20 - y
                 specificgaps = lambda x, y: (-2 - y)
                 alignment = pairwise2.align.globalmc(model_proteins_for_alignment[enzyme], seq_record.seq, 1, -1, fewgaps, specificgaps)
                 fragment_matrix_for_record=fragment_alignment (alignment[0],splitting_lists[enzyme],fastas_aligned_before)
                 #fragment matrix wird die sekundärstruktur normal abgeben
                 fragment_matrix=fragment_matrix.append(fragment_matrix_for_record, ignore_index = True)
                 seq_record_ids=seq_record_ids+[seq_record.id]
        ##obtain features from each fragment
        fragment_matrix_to_pandas_dataframe = pd.DataFrame(fragment_matrix)
        fragments_list = fragment_matrix_to_pandas_dataframe.values.tolist() #matrix is transformed into list

        fragments_length_checker = np.vectorize(len) #counts lengths of fragments
        fragment_matrix_len = fragments_length_checker(fragments_list)
        print(fragment_matrix_len) #returns matrix with the lengths of the fragments

        fragment_matrix_len_to_list = fragment_matrix_len.tolist() #matrix with lenghts of fragments is transformed into list

        list_not_added_up = fragment_matrix_len_to_list
        list_added_up = []        # neue leere Liste
        summe = list_not_added_up[0:]   # summe ist erstes Listenelement
        #Liste ab Element 1 soll jeweils aufaddiert werden
        for list_of_fragments in list_not_added_up:   # Schleife ab dem 2. Element
            inner_list = []
            inner_list.append(list_of_fragments[0])
            summe = list_of_fragments[0]
            for fragment_length in list_of_fragments[1:]:
               summe += fragment_length      # Aufsummieren
               inner_list.append(summe)
            list_added_up.append(inner_list)         # an die neue Liste hängen
        print(list_added_up)

        added_up_frame = pd.DataFrame(list_not_added_up)
        added_up_frame_reindexed = added_up_frame.set_index(pd.Index(fragment_matrix.index)).set_axis(fragments[enzyme], axis=1, inplace=False) #fügt in die added_up_frame wieder den Index ein, der in der ursprünglichen fragment_matrix vorhanden war
        aa_fragment_matrix=pd.DataFrame(columns=fragments[enzyme]) #creates new dataframe
        for index, row in added_up_frame_reindexed.iterrows(): #für jede Reihe des Index, gehe durch die Reihen und wiederhole
            print(row.values, index)
            header = index#erstellt leere Liste mit den jeweiligen headern der Liste und macht sie zu index
            aa_fragments=map_SS8_to_aa(header, row.values.tolist(), filename_AA)
            aa_fragment_matrix=aa_fragment_matrix.append(pd.Series(aa_fragments, index=aa_fragment_matrix.columns), ignore_index=True)
        aa_fragment_matrix=aa_fragment_matrix.set_index(pd.Index(fragment_matrix.index)).set_axis(fragments[enzyme], axis=1, inplace=False)
        print(aa_fragment_matrix.head())
        feature_matrix=featurize(aa_fragment_matrix, permutations, fragments[enzyme], include_charge_features)
        print(feature_matrix.head())
        #set true type of BGC for each BGC as target
        feature_matrix["target"] = BGC_types[BGC_index]
        complete_feature_matrix=complete_feature_matrix.append(feature_matrix, ignore_index = True)


    complete_feature_matrix.to_csv(path_complete_feature_matrix, index=False)
