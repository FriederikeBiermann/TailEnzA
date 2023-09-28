import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Align.Applications import MuscleCommandline
from Bio import AlignIO
from feature_generation import *
from Bio import pairwise2
import pickle
import argparse

# create a parser object
parser = argparse.ArgumentParser(
    description="TailEnzA extracts Genbank files which contain potential novel RiPP biosynthesis gene clusters.")

parser.add_argument("-i", "--input", type=str, nargs=1,
                    metavar="directory_name", default=None,
                    help="Opens and reads the specified folder which contains Genbank files of interest.", required=True)

parser.add_argument("-o", "--output", type=str, nargs=1,
                    metavar="directory_name", default="Output/",
                    help="Output directory")

parser.add_argument("-f", "--frame_length", type=int, nargs=1,
                    metavar="boundary", default=30000,
                    help="determines frame size of the extracted gene window that contains potential novel RiPP BGC") 

parser.add_argument("-t", "--trailing_window", type=int, nargs=1,
                    metavar="boundary", default=5000,
                    help="determines trailing window size of the extracted gene window")                   

args = parser.parse_args()
input = args.input[0]
frame_length = args.frame_length
#print(args.trailing_window)
trailing_window = int(args.trailing_window[0])

try:
    os.mkdir(args.output[0])
except:
    print("WARNING: output directory already existing and not empty.")

muscle = r"muscle"
directory_of_classifiers_BGC_type = "Classifier/BGC_type_affiliation/muscle5_super5_command_BGC_affiliation_alignment_dataset/"
directory_of_classifiers_NP_affiliation = "Classifier/NP_vs_non_NP_affiliation/muscle5_super5_command_NP_vs_non_NP_Classifiers/"
fastas_aligned_before = True
permutation_file = "permutations.txt"
#enzymes=["p450","ycao","Methyl","SAM"]
enzymes=["p450"]
BGC_types=["ripp","nrp","pk"]
include_charge_features=True



with open(permutation_file, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]
# the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"p450":"MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
                              "ycao": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
                              "SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL", 
                              "Methyl": "MGSSHHHHHHSSGLVPRGSHMTTETTTATATAKIPAPATPYQEDIARYWNNEARPVNLRLGDVDGLYHHHYGIGPVDRAALGDPEHSEYEKKVIAELHRLESAQAEFLMDHLGQAGPDDTLVDAGCGRGGSMVMAHRRFGSRVEGVTLSAAQADFGNRRARELRIDDHVRSRVCNMLDTPFDKGAVTASWNNESTMYVDLHDLFSEHSRFLKVGGRYVTITGCWNPRYGQPSKWVSQINAHFECNIHSRREYLRAMADNRLVPHTIVDLTPDTLPYWELRATSSLVTGIEKAFIESYRDGSFQYVLIAADRV"
                              }
start=0
end=350

# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]],
                 "ycao": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 "Methyl":[["begin",0,78],["SAM1",79,104],["SAM2",105,128],["SAM3",129,158],["SAM4",159,188],["SAM5",189,233],["end",234,255]]
                 }
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"],
           "ycao":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "SAM":["begin","SAM","bridging","end"],
           "Methyl":["begin","SAM1","SAM2","SAM3","SAM4","SAM5","end"]
           }
# Classifier, die für die jeweiligen Enzyme die besten balanced accuracy scores erzielt haben
classifiers_enzymes = {"p450":"_AdaBoostClassifier_classifier.sav",
               "ycao":"_ExtraTreesClassifier_classifier.sav",
               "SAM":"_ExtraTreesClassifier_classifier.sav",
               "Methyl":"_ExtraTreesClassifier_classifier.sav"
               }

# classifier_NP_affiliation 
dict_classifier_NP_affiliation = {"p450":"_ExtraTreesClassifier_classifier.sav",
               "ycao":"_AdaBoostClassifier_classifier.sav",
               "SAM":"_AdaBoostClassifier_classifier.sav",
               "Methyl":"_ExtraTreesClassifier_classifier.sav"
              }

def extract_properties(feature):
    
    sequence = feature.qualifiers['translation'][0]
    products = feature.qualifiers['product'][0]
    cds_start = int(feature.location.start)
    if cds_start > 0 :
        cds_start = cds_start + 1
    cds_end = int(feature.location.end)
    return {"sequence": sequence, "product": products, "cds_start": cds_start, "cds_end": cds_end}


def build_dataframe(data_dict, enzyme_name):
    """Builds and returns a DataFrame from a given dictionary."""
    if data_dict:
        df = pd.DataFrame(data_dict).transpose()
        df.insert(0, "Enzyme", enzyme_name)
    else:
        df = pd.DataFrame(
            columns=["sequence", "product", "cds_start", "cds_end"])
        df.insert(0, "Enzyme", enzyme_name)
    return df


def process_records(input_dir):
    """Processes genbank records from files in the given directory."""
    results_dict = {}

    for filename in os.listdir(input_dir):
        if "gb" in filename:
            filepath = os.path.join(input_dir, filename)

            try:
                for gb_record in SeqIO.parse(filepath, "genbank"):
                    process_single_record(gb_record, results_dict)
            except Exception as e:
                print(f"Error: File corrupted or Error Occurred: {e}")

    return results_dict


def process_single_record(gb_record, results_dict):
    """Processes a single genbank record."""
    p450_dict = {}
    methyl_dict = {}
    radical_sam_dict = {}
    ycao_dict = {}

    for i, feature in enumerate(gb_record.features):
        if feature.type == 'CDS' and 'product' in feature.qualifiers:
            product = feature.qualifiers['product'][0].lower()
            id = feature.qualifiers.get(
                'locus_tag', [feature.qualifiers.get('protein_id', ['Unknown'])[0]])[0]

            try:
                if "radical sam" in product:
                    radical_sam_dict[id] = extract_properties(feature)
                elif "p450" in product:
                    p450_dict[id] = extract_properties(feature)
                elif "methyltransferase" in product:
                    methyl_dict[id] = extract_properties(feature)
                elif "ycao" in product:
                    ycao_dict[id] = extract_properties(feature)
            except Exception as e:
                print(f"Error in processing feature: {e}")

    radical_sam_df = build_dataframe(radical_sam_dict, "SAM")
    p450_df = build_dataframe(p450_dict, "p450")
    methyl_df = build_dataframe(methyl_dict, "Methyl")
    ycao_df = build_dataframe(ycao_dict, "ycao")

    # Combine individual DataFrames into a complete DataFrame
    complete_dataframe = pd.concat(
        [radical_sam_df, p450_df, methyl_df, ycao_df], axis=0)

    # Perform classifications, alignments, predictions, and scoring on the complete_dataframe
    # Implement your classification, alignment, prediction, and scoring logic here.

    # For example:
    # complete_dataframe = classify_sequences(complete_dataframe)
    # complete_dataframe = align_sequences(complete_dataframe)
    # complete_dataframe = predict_scores(complete_dataframe)

    # Store the processed records in the results_dict with appropriate keys
    results_dict[gb_record.id] = complete_dataframe


def classify_sequences(df):
    """Classifies the sequences in the DataFrame."""
    # Implement your classification logic here and return the modified DataFrame
    return df


def align_sequences(df):
    """Aligns the sequences in the DataFrame."""
    # Implement your alignment logic here and return the modified DataFrame
    return



results_dataframe = pd.DataFrame(results_dict_row)
results_dataframe = results_dataframe.transpose()
results_dataframe.to_csv(args.output[0]+"results_extracted_genbank_files_metatable.csv", index=False)           
print (results_dataframe)
    
        