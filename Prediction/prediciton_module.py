# Importing modules and python files
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

# Creating ArgumentParser object 
parser = argparse.ArgumentParser(description="TailEnzA extracts Genbank files which contain potential novel RiPP biosynthesis gene clusters.")

# Adding command-line arguments to parser

# -i or --input argument: specifies the input directory containing Genbank files of interest
parser.add_argument("-i", "--input", type=str, nargs=1, metavar="directory_name", default=None,
                    help="Opens and reads the specified folder which contains Genbank files of interest.", required=True)

# -o or --output argument: specifies the output directory for the extracted gene clusters
parser.add_argument("-o", "--output", type=str, nargs=1, metavar="directory_name", default="Output/",
                    help="Output directory")

# -f or --frame_length argument: determines the frame size of the extracted gene window containing potential novel RiPP BGC
parser.add_argument("-f", "--frame_length", type=int, nargs=1, metavar="boundary", default=30000,
                    help="Determines frame size of the extracted gene window that contains potential novel RiPP BGC") 

# -t or --trailing_window argument: determines the trailing window size that is adjacent to the extracted gene window
parser.add_argument("-t", "--trailing_window", type=int, nargs=1, metavar="boundary", default=5000,
                    help="Determines trailing window size of the extracted gene window")

# Parsing the command-line arguments
args = parser.parse_args()

# Extracting the values of the command-line arguments
input = args.input[0]  # Input directory
frame_length = args.frame_length  # Frame size of the extracted gene window
trailing_window = int(args.trailing_window[0]) # Converting the value of the trailing_window argument to an integer

# Create an output directory if it doesn't exist
try:
    os.mkdir(args.output[0])
except:
    print("WARNING: output directory already existing and not empty.")

muscle = r"muscle" # Path to the muscle5 executable tool
directory_of_classifiers_BGC_type = "Classifier/BGC_type_affiliation/muscle5_super5_command_BGC_affiliation_alignment_dataset/" # Directory that contains classifiers for BGC type affiliation
directory_of_classifiers_NP_affiliation = "Classifier/NP_vs_non_NP_affiliation/muscle5_super5_command_NP_vs_non_NP_Classifiers/" # Directory that contains classifiers for natural product vs. non-natural product BGC affiliation
fastas_aligned_before = True
permutation_file = "permutations.txt" # File containing the permutations sequences
#enzymes=["p450"] 
enzymes=["p450","ycao","Methyl","SAM"] # List of tailoring enzymes
BGC_types=["ripp","nrp","pk"] # List of BGC types to consider
include_charge_features=True

with open(permutation_file, 'r') as file: # Read permutations from the file and store them in a list
    permutations = [line.rstrip('\n') for line in file]

# Model tailoring enzymes for alignment that are used as reference
model_proteins_for_alignment = {
    "p450": "MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
    "ycao": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
    "SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL", 
    "Methyl": "MGSSHHHHHHSSGLVPRGSHMTTETTTATATAKIPAPATPYQEDIARYWNNEARPVNLRLGDVDGLYHHHYGIGPVDRAALGDPEHSEYEKKVIAELHRLESAQAEFLMDHLGQAGPDDTLVDAGCGRGGSMVMAHRRFGSRVEGVTLSAAQADFGNRRARELRIDDHVRSRVCNMLDTPFDKGAVTASWNNESTMYVDLHDLFSEHSRFLKVGGRYVTITGCWNPRYGQPSKWVSQINAHFECNIHSRREYLRAMADNRLVPHTIVDLTPDTLPYWELRATSSLVTGIEKAFIESYRDGSFQYVLIAADRV"
                              }
start=0
end=350

# Functional fragments of tailoring enzymes they will be cut into
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]],
                 "ycao": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 "Methyl":[["begin",0,78],["SAM1",79,104],["SAM2",105,128],["SAM3",129,158],["SAM4",159,188],["SAM5",189,233],["end",234,255]]
                 }
# Description of the respective functional fragments in tailoring enzymes
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"],
           "ycao":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "SAM":["begin","SAM","bridging","end"],
           "Methyl":["begin","SAM1","SAM2","SAM3","SAM4","SAM5","end"]
}

# Dictionary mapping enzymes to their respective best classifiers for BGC type affiliation
classifiers_enzymes = {
    "p450": "_AdaBoostClassifier_classifier.sav",
    "ycao": "_ExtraTreesClassifier_classifier.sav",
    "SAM": "_ExtraTreesClassifier_classifier.sav",
    "Methyl": "_ExtraTreesClassifier_classifier.sav"
}

# Dictionary mapping enzymes to their respective best classifiers for NP vs. non-NP affiliation
dict_classifier_NP_affiliation = {
    "p450": "_ExtraTreesClassifier_classifier.sav",
    "ycao": "_AdaBoostClassifier_classifier.sav",
    "SAM": "_AdaBoostClassifier_classifier.sav",
    "Methyl": "_ExtraTreesClassifier_classifier.sav"
}

# Function to extract feature properties
def extract_feature_properties(feature):
    # Extracting the translated gene sequence of a the four enzyme types, their product, coding region start (cds_start), and coding region end (cds_end) from the Genbank file.
    sequence = feature.qualifiers['translation'][0]
    products = feature.qualifiers['product'][0]
    cds_start = int(feature.location.start)
    if cds_start > 0:
        cds_start = cds_start + 1 
    cds_end = int(feature.location.end)
    
    # Return the dictionary with the extracted features
    return {"sequence": sequence, "product": products, "cds_start": cds_start, "cds_end": cds_end}

# Create an empty dictionary to store the results for all extracted tailoring enzymes
results_dict_row = {}
for filename in os.listdir(input): # Iterate over the files in the input directory
    if "gb" in filename: # Check if the file has the "gb" extension
        f = os.path.join(input, filename) # Create the full path to the file

# Iterate over the GenBank records in the file
        for gb_record in SeqIO.parse(f, "genbank"):
            try:
                gb_feature = gb_record.features
            # Create dictionaries to store feature properties for the different tailoring enzyme types
                p450_dict = {}
                methyl_dict = {}
                radical_sam_dict = {}
                ycao_dict= {}
                id_list = []
                for i, feature in enumerate(gb_record.features):  # Iterate over the features in the GenBank record
                    if feature.type == 'CDS': # Check if the feature type is 'CDS'
                        if 'product' in feature.qualifiers: # Check if the 'product' qualifier is present to verify it codes for a real protein
                            product = feature.qualifiers['product'][0].lower()
                            
                            if "radical sam" in product:  # If 'radical sam' is in the 'product' qualifier extract the locus tag or protein ID as the identifier
                                try:
                                    id = feature.qualifiers['locus_tag'][0]
                                except:
                                    id = feature.qualifiers['protein_id'][0]
                                
                                try: # Continue with creating the dictionary for the respective radical SAM using the function 'extract_feature_properties'
                                    radical_sam_dict[id] = extract_feature_properties(feature)
                                except:
                                    continue
                            
                            elif "p450" in product: # If 'p450' is in the 'product' qualifier extract the locus tag or protein ID as the identifier
                                try:
                                    id = feature.qualifiers['locus_tag'][0]
                                except:
                                    id = feature.qualifiers['protein_id'][0]
                                    p450_dict[id] = extract_feature_properties(feature)
                                
                                try: 
                                    p450_dict[id] = extract_feature_properties(feature)
                                except:
                                    continue

                            elif "methyltransferase" in product: # If 'methyltransferase' is in the 'product' qualifier extract the locus tag or protein ID as the identifier
                                try:
                                    id = feature.qualifiers['locus_tag'][0]
                                except:
                                    id = feature.qualifiers['protein_id'][0]
                                try:
                                    methyl_dict[id] = extract_feature_properties(feature)
                                except:
                                    continue

                            elif "ycao" in product: # If 'ycao' is in the 'product' qualifier extract the locus tag or protein ID as the identifier
                                try:
                                    id = feature.qualifiers['locus_tag'][0]
                                except:
                                    id = feature.qualifiers['protein_id'][0]
                                try:
                                    ycao_dict[id] = extract_feature_properties(feature)     
                                except:
                                    continue
                
                if radical_sam_dict: # Check if radical_sam_dict is not empty
                    radical_sam_df = pd.DataFrame(radical_sam_dict) # if it is not empty, create a DataFrame from 'radical_sam_dict' and transpose it
                    radical_sam_df = radical_sam_df.transpose()
                    radical_sam_df.insert(0, "Enzyme", "SAM") # Insert "Enzyme" column with the value "SAM" at the beginning of the DataFrame
                else: # If there is no 'radical_sam_dict', create an empty DataFrame with specified columns and insert "Enzyme" column with the value "SAM"
                    radical_sam_df = pd.DataFrame(columns=["sequence", "product", "cds_start", "cds_end"])
                    radical_sam_df.insert(0, "Enzyme", "SAM")

                # Does this step for each tailoring enzymes (Cytochrome P450, Methyltransferase, YcaO) now
                if p450_dict:
                    p450_df = pd.DataFrame(p450_dict)
                    p450_df = p450_df.transpose()
                    p450_df.insert(0, "Enzyme", "p450")
                else:
                    p450_df = pd.DataFrame(columns=["sequence", "product", "cds_start", "cds_end"])
                    p450_df.insert(0, "Enzyme", "p450")

                if methyl_dict:
                    methyl_df = pd.DataFrame(methyl_dict)
                    methyl_df = methyl_df.transpose()
                    methyl_df.insert(0, "Enzyme", "Methyl")
                else:
                    methyl_df = pd.DataFrame(columns=["sequence", "product", "cds_start", "cds_end"])
                    methyl_df.insert(0, "Enzyme", "Methyl")

                if ycao_dict:
                    ycao_df = pd.DataFrame(ycao_dict)
                    ycao_df = ycao_df.transpose()
                    ycao_df.insert(0, "Enzyme", "ycao")
                else:
                    ycao_df = pd.DataFrame(columns=["sequence", "product", "cds_start", "cds_end"])
                    ycao_df.insert(0, "Enzyme", "ycao")

                # Concatenate the four single dataframes into a complete_dataframe 
                complete_dataframe = pd.concat([ycao_df, p450_df, methyl_df, radical_sam_df], axis=0)
                #complete_dataframe = pd.concat([p450_df])
                print(complete_dataframe)

                # Add empty columns to the complete_dataframe
                complete_dataframe["BGC_type"] = ""
                complete_dataframe["BGC_type_score"] = ""
                complete_dataframe["NP_BGC_affiliation"] = ""
                complete_dataframe["NP_BGC_affiliation_score"] = ""
                complete_dataframe["extracted_window_start"] = ""
                complete_dataframe["extracted_window_end"] = ""

                # Calculate window start and end positions based on the cds_start and frame_length
                complete_dataframe["extracted_window_start"] = complete_dataframe["cds_start"]
                complete_dataframe["extracted_window_start"] = complete_dataframe["extracted_window_start"].astype('int')
                complete_dataframe["extracted_window_end"] = complete_dataframe["cds_start"] + frame_length
                complete_dataframe["extracted_window_end"] = complete_dataframe["extracted_window_end"].astype('int')

                if len(complete_dataframe) != 0:
                    fragment_rows = []
                    fragment_matrix = pd.DataFrame()
                    enzyme = complete_dataframe["Enzyme"][0] # Get the tailoring enzyme type from the first row of complete_dataframe
                    
                    # Create translated_sequences list from the extracted tailoring enzyme translated sequences, respectively
                    translated_sequences = [SeqRecord(Seq(model_proteins_for_alignment[enzyme]), id="Reference")]
                    for index, row in complete_dataframe.iterrows():
                        translated_sequences.append(SeqRecord(Seq(row["sequence"]), id=index))
                    SeqIO.write(translated_sequences, "temp/temp_input.fasta", "fasta") # Write the translated_sequences list to a temporary fasta file                 
                    # Perform sequence alignment using muscle5
                    muscle_command_line = f"{muscle} -super5 temp/temp_input.fasta -output temp/temp_output.fasta"
                    os.system(muscle_command_line)
                    
                    # Read the aligned sequences from the output file
                    alignment = AlignIO.read(open("temp/temp_output.fasta"), "fasta")
                    
                    # Fragment the alignment based on splitting lists
                    fragment_matrix = fragment_alignment(alignment, splitting_lists[enzyme], fastas_aligned_before)
                    fragment_matrix.set_index(complete_dataframe.index)
                    
                    # Featurize the fragment matrix
                    feature_matrix = featurize(fragment_matrix, permutations, fragments[enzyme], include_charge_features)
                    
                    # Predict NP-BGC affiliation
                    natural_product_classifier = directory_of_classifiers_NP_affiliation + enzyme + dict_classifier_NP_affiliation[enzyme]
                    classifier_NP_affiliation = pickle.load(open(natural_product_classifier, 'rb'))
                    predicted_NP_affiliation = classifier_NP_affiliation.predict(feature_matrix)
                    score_predicted_NP_affiliation = classifier_NP_affiliation.predict_proba(feature_matrix)
                    
                    # Update complete_dataframe with NP-BGC affiliation predictions and scores
                    complete_dataframe["NP_BGC_affiliation"] = predicted_NP_affiliation
                    complete_dataframe["NP_BGC_affiliation_score"] = [max(score_list_NP_affiliation) for score_list_NP_affiliation in score_predicted_NP_affiliation]
                    
                    # Predict BGC type
                    BGC_type_classifier = directory_of_classifiers_BGC_type + enzyme + classifiers_enzymes[enzyme]
                    classifier_BGC_type = pickle.load(open(BGC_type_classifier, 'rb'))
                    predicted_BGC = classifier_BGC_type.predict(feature_matrix)
                    score_predicted_BGCs = classifier_BGC_type.predict_proba(feature_matrix)
                    
                    # Update complete_dataframe with BGC type predictions and scores
                    complete_dataframe["BGC_type"] = predicted_BGC
                    complete_dataframe["BGC_type_score"] = [max(score_list_BGC_type) for score_list_BGC_type in score_predicted_BGCs]
                    
                    # Initialize score_list_NP_affiliation and score_list_BGC_type
                    score_list_NP_affiliation = []
                    score_list_BGC_type = []
                    # Iterate over each row in complete_dataframe
                    for index, row in complete_dataframe.iterrows():
                        window_start = row["extracted_window_start"]
                        window_end = row["extracted_window_end"]
                        
                        # Create a filtered_dataframe that contains rows within the window
                        filtered_dataframe = complete_dataframe[(complete_dataframe['cds_start'] >= window_start) & (complete_dataframe['cds_end'] <= window_end)]
                        print(filtered_dataframe)
                        score = 0 # Setting score to 0
                        
                        # Calculate score for each window based on BGC type and NP-BGC affiliation
                        for index, rows_rows in filtered_dataframe.iterrows():
                            if rows_rows["BGC_type"] == "ripp": # If BGC type is RiPP
                                score += (1 + rows_rows["BGC_type_score"]) * rows_rows["NP_BGC_affiliation_score"]
                                score = round(score, 3)
                            else: # If BGC type is not RiPP
                                score -= (rows_rows["BGC_type_score"] + 1) * rows_rows["NP_BGC_affiliation_score"]
                                score = round(score, 3)
                        
                        # Extract the corresponding sequence record for the window
                        record = gb_record[max(0, window_start - trailing_window):min(window_end + trailing_window, len(gb_record.seq))]
                        record.annotations["molecule_type"] = "dna"
                        record.annotations["score"] = score
                        filename_record = f"{gb_record.id}_{window_start}_{window_end}_{score}.gb"
                        
                        # Write sequence record to an output file if the score is above a threshold
                        if score >= -2:
                            SeqIO.write(record, args.output[0] + filename_record, "gb")
                        
                        # Update results_dict_row with the window details and score
                        results_dict_row[f"{gb_record.id}_{window_start}"] = {"ID": gb_record.id, "description": gb_record.description,"window_start": window_start, "window_end": window_end, "score": score, "filename": filename_record}
            except: 
                print("Error: File corrupted.")

# Create results_dataframe from the results_dict_row dictionary and save it to a new CSV file
results_dataframe = pd.DataFrame(results_dict_row)
results_dataframe = results_dataframe.transpose()
results_dataframe.to_csv(args.output[0] + "results_extracted_genbank_files_metatable.csv", index=False)
print(results_dataframe)



