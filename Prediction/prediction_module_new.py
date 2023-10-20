import os
import sys
import subprocess
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import AlignIO
from feature_generation import *
from enzyme_information import enzymes
import pickle
import argparse
import pyhmmer

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
trailing_window = int(args.trailing_window[0])

directory_of_classifiers_BGC_type = "Classifier/BGC_type_affiliation/muscle5_super5_command_BGC_affiliation_alignment_dataset/"
directory_of_classifiers_NP_affiliation = "Classifier/NP_vs_non_NP_affiliation/muscle5_super5_command_NP_vs_non_NP_Classifiers/"
fastas_aligned_before = True
permutation_file = "permutations.txt"
BGC_types=["ripp","nrp","pk"]
include_charge_features=True

try:
    os.mkdir(args.output[0])
except:
    print("WARNING: output directory already existing and not empty.")

with open(permutation_file, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]

def extract_feature_properties(feature):
    
    sequence = feature.qualifiers['translation'][0]
    products = feature.qualifiers['product'][0]
    cds_start = int(feature.location.start)
    if cds_start > 0 :
        cds_start = cds_start + 1
    cds_end = int(feature.location.end)
    return {"sequence": sequence, "product": products, "cds_start": cds_start, "cds_end": cds_end}

def get_identifier(feature):
    """Returns the 'locus_tag' or 'protein_id' from a feature"""
    return feature.qualifiers.get('locus_tag', [feature.qualifiers.get('protein_id', [None])[0]])[0]

def process_feature_dict(product_dict, enzyme_name):
    """Process the feature dictionary and returns a DataFrame"""
    if product_dict:
        df = pd.DataFrame(product_dict).transpose()
        df.insert(0, "Enzyme", enzyme_name)
    else:
        df = pd.DataFrame(columns=["sequence", "product", "cds_start", "cds_end"])
        df.insert(0, "Enzyme", enzyme_name)
    return df

def set_dataframe_columns(df):
    """Sets default columns to a dataframe"""
    df["BGC_type"] = ""
    df["BGC_type_score"] = ""
    df["NP_BGC_affiliation"] = ""
    df["NP_BGC_affiliation_score"] = ""
    df["30kb_window_start"] = df["cds_start"].astype('int')
    df["30kb_window_end"] = (df["cds_start"] + frame_length).astype('int')
    return df

def muscle_align_sequences(fasta_filename, enzyme):
    """Align sequences using muscle and returns the alignment"""

    muscle_cmd = ["muscle", "-in", fasta_filename, "-out", f"aligned_{fasta_filename}" , "-seqtype", "protein", "-maxiters", "16", "-gapopen", str(enzymes[enzyme].gap_opening_penalty), "-gapextend", str(enzymes[enzyme].gap_extend_penalty)]
    
    try:
        subprocess.check_call(muscle_cmd)
    except subprocess.CalledProcessError:
        print(f"Error: Failed to run command {' '.join(muscle_cmd)}")
        sys.exit(1)

    return AlignIO.read(open(f"aligned_{fasta_filename}"), "fasta")

def create_feature_lookup(record):
    """
    Create a lookup dictionary from the record's features
    Keyed by protein_id (or locus_tag if protein_id is not available)
    """
    feature_lookup = {}
    for feature in record.features:
        if feature.type == "CDS":
            protein_id = feature.qualifiers.get('protein_id', [feature.qualifiers.get('locus_tag', [None])[0]])[0]
            feature_lookup[protein_id] = feature
    return feature_lookup

def run_hmmer(record, enzyme):
    enzyme_hmm_filename = enzymes[enzyme]["hmm_file"]
    fasta = f"{record.id}_temp.fasta"
    genbank_to_fasta_cds(record, fasta)  # Assuming this function is correct

    feature_lookup = create_feature_lookup(record)

    results = []

    if os.path.exists(enzyme_hmm_filename):  # using enzyme_hmm_filename
        with pyhmmer.easel.SequenceFile(fasta, digital=True) as seq_file:
            sequences = seq_file.read_block()
            with pyhmmer.plan7.HMMFile(enzyme_hmm_filename) as hmm_file:
                for hits in pyhmmer.hmmsearch(hmm_file, sequences, cpus=4):
                    for hit in hits:
                        evalue = hit.evalue
                        hit_name = hit.name.decode()

                        if evalue >= 10e-20:
                            continue

                        feature = feature_lookup.get(hit_name)
                        if feature:
                            results.append(extract_feature_properties(feature))
    return results

def genbank_to_fasta_cds(record, fasta_file):
    if os.path.exists(fasta_file):
        return
    with open(fasta_file, "w") as output_handle:
        for feature in record.features:
            if feature.type == "CDS":
                try:
                    # Get the protein ID or locus tag for the sequence ID in the FASTA file
                    protein_id = feature.qualifiers.get('protein_id', [feature.qualifiers.get('locus_tag')[0]])[0]
                    sequence = feature.location.extract(record).seq
                    # Create a new SeqRecord and write to the output handle
                    SeqIO.write(SeqIO.SeqRecord(sequence, id=protein_id, description=""), output_handle, "fasta")
                except Exception as e:
                    print("Error processing feature:", e)
                    continue

def save_enzymes_to_fasta(record_dict):
    # Save enzymes together with reference to fasta 
    for enzyme, results in record_dict.items():
        # Create a SeqRecord for the reference sequence
        reference_record = SeqRecord(Seq(enzymes[enzyme]["reference_for_alignment"]), id="Reference", description="Reference Sequence")

        # Generate a list of SeqRecord objects from the results, with the reference sequence at the beginning
        seq_records = [reference_record] + [
            SeqRecord(Seq(result["sequence"]), id=f"{enzyme}_{result['cds_start']}_{result['cds_end']}", description=result["product"])
            for result in results
        ]
        
        fasta_name = f"{enzyme}_tailoring_enzymes.fasta"
        SeqIO.write(seq_records, fasta_name, "fasta")

def classifier_prediction(feature_matrix, classifier_path):
    """Predict values using a classifier"""
    classifier = pickle.load(open(classifier_path, 'rb'))
    predicted_values = classifier.predict(feature_matrix)
    score_predicted_values = classifier.predict_proba(feature_matrix)
    return predicted_values, score_predicted_values

def process_dataframe_and_save(complete_dataframe, gb_record, trailing_window, output_path):
    results_dict = {}

    for index, row in complete_dataframe.iterrows():
        window_start = row["30kb_window_start"]
        window_end = row["30kb_window_end"]
        
        # Filter dataframe based on window
        filtered_dataframe = complete_dataframe[(complete_dataframe['cds_start'] >= window_start) & (complete_dataframe['cds_end'] <= window_end)]
        
        # Compute score for filtered dataframe
        score = 0
        for _, rows_rows in filtered_dataframe.iterrows():
            if rows_rows["BGC_type"]=="ripp":
                score += (1 + rows_rows["BGC_type_score"])*rows_rows["NP_BGC_affiliation_score"]
            else:
                score -= (rows_rows["BGC_type_score"] + 1)*rows_rows["NP_BGC_affiliation_score"]
            score = round(score, 3)
        
        # Extract record based on window and score
        record = gb_record[max(0, window_start-trailing_window):min(window_end+trailing_window, len(gb_record.seq))]
        record.annotations["molecule_type"] = "dna"
        record.annotations["score"] = score
        filename_record = f"{gb_record.id}_{window_start}_{window_end}_{score}.gb"
        
        if score >= -2:
            SeqIO.write(record, output_path + filename_record, "gb")
        
        results_dict[f"{gb_record.id}_{window_start}"] = {
            "ID": gb_record.id,
            "description": gb_record.description,
            "window_start": window_start,
            "window_end": window_end,
            "score": score,
            "filename": filename_record
        }
        
    return results_dict

for filename in os.listdir(input):

    if "gb" in filename:
        file_path = os.path.join(input, filename)
        for gb_record in SeqIO.parse(file_path, "genbank"):
            # Create datastructure for results and fill with hmmer results
            tailoring_enzymes_in_record = {key:{run_hmmer(gb_record, key)} for key in enzymes}
            enzyme_dataframes = {enzyme_name: set_dataframe_columns(process_feature_dict(enzyme_dict, enzyme_name)) for enzyme_name, enzyme_dict in tailoring_enzymes_in_record.items()}
            complete_dataframe = pd.concat([enzyme_dataframe for enzyme_dataframe in enzyme_dataframes.values], axis=0)
            # Save enzymes together with reference to fasta for running the alignment on it
            save_enzymes_to_fasta(tailoring_enzymes_in_record, enzymes)
            fasta_dict = {key: f"{key}_tailoring_enzymes.fasta" for key in enzymes}
            alignments = {enzyme: muscle_align_sequences(filename, enzyme) for enzyme, filename in fasta_dict.items()}
            #TODO: fragment to new splitting list, set index
            fragment_matrixes = {key: fragment_alignment(alignments[key],enzymes[key].splitting_list,fastas_aligned_before) for key in enzymes} 
            feature_matrixes = {key: featurize(fragment_matrix, permutations, enzymes[key].splitting_list.keys, include_charge_features) for key, fragment_matrix in fragment_matrixes.items()}
            #TODO: set index correctly
            # Load the classifiers
            classifiers_metabolism = {key: pickle.load(open(directory_of_classifiers_NP_affiliation+key+enzymes[key].classifier_metabolism, "rb")) for key in enzymes}
            predicted_metabolisms = {key: classifiers_metabolism[key].predict(feature_matrix) for key, feature_matrix in feature_matrixes.items()}
            scores_predicted_metabolism = {key: classifiers_metabolism[key].predict_proba(feature_matrix) for key, feature_matrix in feature_matrixes.items()}
            
            classifiers_BGC_type = {key: pickle.load(open(directory_of_classifiers_BGC_type+key+enzymes[key].classifier_BGC_type, "rb")) for key in enzymes}
            predicted_BGC_types = {key: classifiers_BGC_type[key].predict(feature_matrix) for key, feature_matrix in feature_matrixes.items()}
            scores_predicted_BGC_type = {key: classifiers_BGC_type[key].predict_proba(feature_matrix) for key, feature_matrix in feature_matrixes.items()}

            for enzyme in enzymes:
    
                # Create a dictionary mapping for each predicted value
                predicted_metabolism_dict = dict(zip(enzyme_dataframes[enzyme].index, predicted_metabolisms[enzyme]))
                score_predicted_metabolism_dict = dict(zip(enzyme_dataframes[enzyme].index, [scores[prediction] for prediction, scores in zip(predicted_metabolisms[enzyme], scores_predicted_metabolism[enzyme])]))

                predicted_BGC_type_dict = dict(zip(enzyme_dataframes[enzyme].index, predicted_BGC_types[enzyme]))
                score_predicted_BGC_type_dict = dict(zip(enzyme_dataframes[enzyme].index, [scores[prediction] for prediction, scores in zip(predicted_BGC_types[enzyme], scores_predicted_BGC_type[enzyme])]))

                # Map the predictions and scores to the dataframe using the dictionaries
                complete_dataframe["NP_BGC_affiliation"] = complete_dataframe.index.map(predicted_metabolism_dict).fillna(complete_dataframe["NP_BGC_affiliation"])
                complete_dataframe["NP_BGC_affiliation_score"] = complete_dataframe.index.map(score_predicted_metabolism_dict).fillna(complete_dataframe["NP_BGC_affiliation_score"])
                
                complete_dataframe["BGC_type"] = complete_dataframe.index.map(predicted_BGC_type_dict).fillna(complete_dataframe["BGC_type"])
                complete_dataframe["BGC_type_score"] = complete_dataframe.index.map(score_predicted_BGC_type_dict).fillna(complete_dataframe["BGC_type_score"])
            
            results = process_dataframe_and_save(complete_dataframe, gb_record, trailing_window, args.output[0])