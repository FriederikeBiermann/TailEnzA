
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from TailEnzA.Prediction.enzyme_prediction import * 
from TailEnzA.Classifiers.feature_generation import *
from Bio import pairwise2
file_for_prediction="Input/Testfile.fasta"
directory_of_classifiers="Classifier/Toy_dataset/"
fastas_aligned_before=False
permutation_file="permutations.txt"
output_directory="Output/"
filename_output=output_directory+file_for_prediction.split("/")[1].split(".")[0]+".csv"
enzymes=["p450","YCAO","SAM"]
BGC_types=["ripp","nrp","pk"]
include_charge_features=False

with open(permutation_file, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]
# the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"p450":"MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
                              "YCAO": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",  
                              "SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL"}
start=0
end=350
# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]], 
                 "YCAO": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]]}
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"], 
           "YCAO":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "SAM":["begin","SAM","bridging","end"]
           }
results= pd.DataFrame()
for record in SeqIO.parse(file_for_prediction, "fasta"):
        # predict enzyme-type of given fasta
        enzyme=enzyme_calculation(str(record.seq))[0]
        fragment_matrix=pd.DataFrame()
        # align the AA sequence against the model protein for that enzyme type and fragment according to the splitting list
        fewgaps = lambda x, y: -20 - y
        specificgaps = lambda x, y: (-2 - y)
        alignment = pairwise2.align.globalmc(model_proteins_for_alignment[enzyme], record.seq, 1, -1, fewgaps, specificgaps)
        fragment_matrix=fragment_alignment (alignment[0],splitting_lists[enzyme],fastas_aligned_before)
        feature_matrix=featurize(fragment_matrix, permutations, fragments[enzyme], include_charge_features)
        print (feature_matrix)
        used_classifier=directory_of_classifiers+enzyme+"_ExtraTreesClassifier_classifier.sav"
        classifier = pickle.load(open(used_classifier, 'rb'))
        predicted_BGC=classifier.predict(feature_matrix)
        score_predicted_BGCs = classifier.predict_proba(feature_matrix)
        new_row={"Record":record.id, "Predicted enzyme": enzyme, "Predicted BGC affiliation": predicted_BGC, "Score": score_predicted_BGCs}
        results = results.append(new_row, ignore_index=True)

results.to_csv(filename_output, index=False) 
