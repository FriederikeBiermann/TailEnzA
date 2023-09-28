
import Bio
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from feature_generation import *
# if fastas aligned before True-> use alignment made for instance in geneious utilizing MUSCLE align -> best for larger datasets
fastas_aligned_before=True
# if include_charge_features=True-> features describing general electrostatic information will be included
include_charge_features=True
#fill in filenames here!
foldernameoutput="Classifiers/Toy_dataset/"
foldername_training_sets="Training_data/Toy_dataset/toy_"
filename_permutations="permutations.txt"

enzymes=["p450","YCAO","SAM","Methyl"]
BGC_types=["ripp","nrp","pk"]
# the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"p450":"MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
                              "YCAO": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
                              "SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL",
                              "Methyl":"MGSSHHHHHHSSGLVPRGSHMTTETTTATATAKIPAPATPYQEDIARYWNNEARPVNLRLGDVDGLYHHHYGIGPVDRAALGDPEHSEYEKKVIAELHRLESAQAEFLMDHLGQAGPDDTLVDAGCGRGGSMVMAHRRFGSRVEGVTLSAAQADFGNRRARELRIDDHVRSRVCNMLDTPFDKGAVTASWNNESTMYVDLHDLFSEHSRFLKVGGRYVTITGCWNPRYGQPSKWVSQINAHFECNIHSRREYLRAMADNRLVPHTIVDLTPDTLPYWELRATSSLVTGIEKAFIESYRDGSFQYVLIAADRV"}
start=0
end=350
# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]],
                 "YCAO": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 "Methyl":[["begin",0,78],["SAM1",79,104],["SAM2",105,128],["SAM3",129,158],["SAM4",159,188],["SAM5",189,233],["end",234,255]]}
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"],
           "YCAO":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "SAM":["begin","SAM","bridging","end"],
           "Methyl":["begin","SAM1","SAM2","SAM3","SAM4","SAM5","end"]
           }


#permutations= list of 4-aa motifs to use as features
with open(filename_permutations, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]
def create_filenames(enzymes, BGC_types,foldername_training_sets):

    #creates a list of all filenames of the input files -> change the exact expression if files named differently

    filenames=[]
    if fastas_aligned_before==False:
        for BGC in BGC_types:
            for enzyme in enzymes:
                filenames+=[foldername_training_sets+BGC+"_"+enzyme+".fasta"]
    if fastas_aligned_before==True:
        for BGC in BGC_types:
            for enzyme in enzymes:
                filenames+=[foldername_training_sets+BGC+"_"+enzyme+" alignment.fas"]
    return filenames
for enzyme in enzymes:
    # the complete feature matrix is the input for all subsequent steps-> as we have different classifiers for each enzyme, we also need different training sets for each enzyme
    complete_feature_matrix=pd.DataFrame()
    path_complete_feature_matrix=foldernameoutput+enzyme+"_complete_feature_matrix.csv"
    filenames_training_set=create_filenames([enzyme],BGC_types,foldername_training_sets)

    for BGC_index, dataset in enumerate(filenames_training_set):
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
                 fragment_matrix=fragment_matrix.append(fragment_matrix_for_record, ignore_index = True)
                 seq_record_ids=seq_record_ids+[seq_record.id]
        #obtain features from each fragment
        feature_matrix=featurize(fragment_matrix, permutations, fragments[enzyme], include_charge_features)
        print (feature_matrix)
        #set true type of BGC for each BGC as target
        feature_matrix["target"] = BGC_types[BGC_index]
        complete_feature_matrix=complete_feature_matrix.append(feature_matrix, ignore_index = True)

    complete_feature_matrix.to_csv(path_complete_feature_matrix, index=False)
