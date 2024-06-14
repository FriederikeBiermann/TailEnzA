import os
import sys
from sklearn.preprocessing import LabelEncoder
import subprocess
import shutil
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import AlignIO
from tailenza.data.enzyme_information import enzymes, BGC_types
import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pyhmmer
import warnings
import esm
import logging
from importlib.resources import files
from Bio import BiopythonWarning
from subprocess import DEVNULL
from tailenza.classifiers.preprocessing.feature_generation import (
    fragment_alignment,
    featurize_fragments,
)

import logging
from importlib.resources import files
import esm
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch import nn

from tailenza.classifiers.machine_learning.machine_learning_training_classifiers_AA_BGC_type import (
    LSTM,
    BasicFFNN,
    IntermediateFFNN,
    AdvancedFFNN,
    VeryAdvancedFFNN,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore", message="is_sparse is deprecated and will be removed in a future version."
)
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create a parser object
parser = argparse.ArgumentParser(
    description="TailEnzA extracts Genbank files which contain potential unconventional biosynthesis gene clusters."
)

parser.add_argument(
    "-i",
    "--input",
    type=str,
    nargs=1,
    metavar="file_name",
    default=None,
    help="Opens and reads the specified genbank file.",
    required=True,
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    nargs=1,
    metavar="directory_name",
    default="Output/",
    help="Output directory",
)


parser.add_argument(
    "-c",
    "--score_cutoff",
    type=float,
    nargs=1,
    metavar="cutoff",
    default=0.75,
    help="Cutoff score to use for the genbank extraction.",
)

args = parser.parse_args()
directory_of_classifiers_BGC_type = "../classifiers/classifiers/BGC_type/"
directory_of_classifiers_NP_affiliation = "../classifiers/classifiers/metabolism_type/"
fastas_aligned_before = True
include_charge_features = True
package_dir = files("tailenza").joinpath("")
hmm_dir = package_dir.joinpath("data", "HMM_files")


def extract_feature_properties(feature):

    sequence = feature.qualifiers["translation"][0]
    products = feature.qualifiers.get("product", ["Unknown"])[0]
    cds_start = int(feature.location.start)
    if cds_start > 0:
        cds_start = cds_start + 1
    cds_end = int(feature.location.end)
    return {
        "sequence": sequence,
        "product": products,
        "cds_start": cds_start,
        "cds_end": cds_end,
    }


def get_identifier(feature):
    """Returns the 'locus_tag' or 'protein_id' from a feature"""
    return feature.qualifiers.get(
        "protein_id", feature.qualifiers.get("locus_tag", ["Unknown"])
    )[0]


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
    return df


def muscle_align_sequences(fasta_filename, enzyme):
    """Align sequences using muscle and returns the alignment"""

    # Check if the fasta file contains only the reference sequence
    num_sequences = sum(1 for _ in SeqIO.parse(fasta_filename, "fasta"))
    if num_sequences <= 1:
        # If the file contains only the reference, read it in as an alignment
        return AlignIO.read(open(fasta_filename), "fasta")

    muscle_cmd = [
        "muscle",
        "-in",
        fasta_filename,
        "-out",
        f"{fasta_filename}_aligned.fasta",
        "-seqtype",
        "protein",
        "-maxiters",
        "16",
        "-gapopen",
        str(enzymes[enzyme]["gap_opening_penalty"]),
        "-gapextend",
        str(enzymes[enzyme]["gap_extend_penalty"]),
    ]

    try:
        subprocess.check_call(muscle_cmd, stdout=DEVNULL, stderr=DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Error: Failed to run command {' '.join(muscle_cmd)}")
        sys.exit(1)

    return AlignIO.read(open(f"{fasta_filename}_aligned.fasta"), "fasta")


def create_feature_lookup(record):
    """
    Create a lookup dictionary from the record's features
    Keyed by protein_id (or locus_tag if protein_id is not available)
    """
    feature_lookup = {}
    for feature in record.features:
        if feature.type == "CDS":
            protein_id = feature.qualifiers.get(
                "protein_id", feature.qualifiers.get("locus_tag", ["Unknown"])
            )[0]
            feature_lookup[protein_id] = feature
    return feature_lookup


def run_hmmer(record, enzyme):
    try:
        enzyme_hmm_filename = os.path.join(hmm_dir, enzymes[enzyme]["hmm_file"])
        fasta = os.path.join(tmp_dir, f"{record.id[:-2]}_temp.fasta")
    except KeyError as e:
        logging.error(f"Key error: {e}")
        return {}

    try:
        genbank_to_fasta_cds(record, fasta)
    except Exception as e:
        logging.error(f"Error in converting GenBank to FASTA: {e}")
        return {}

    try:
        feature_lookup = create_feature_lookup(record)
    except Exception as e:
        logging.error(f"Error in creating feature lookup: {e}")
        return {}

    results = {}

    if not os.path.exists(enzyme_hmm_filename):
        logging.error(f"HMM file {enzyme_hmm_filename} does not exist.")
        return results

    try:
        with pyhmmer.easel.SequenceFile(fasta, digital=True) as seq_file:
            sequences = seq_file.read_block()
            with pyhmmer.plan7.HMMFile(enzyme_hmm_filename) as hmm_file:
                hmm = hmm_file.read()

                try:
                    pipeline = pyhmmer.plan7.Pipeline(hmm.alphabet)
                    for hit in pipeline.search_hmm(hmm, sequences):

                        evalue = hit.evalue
                        hit_name = hit.name.decode()
                        if evalue >= 10e-10:
                            continue

                        feature = feature_lookup.get(hit_name)
                        if feature:
                            results[get_identifier(feature)] = (
                                extract_feature_properties(feature)
                            )
                except Exception as e:
                    logging.error(f"Error during HMMER search: {e}")
    except Exception as e:
        logging.error(f"Error opening or processing files: {e}")
    return results


def genbank_to_fasta_cds(record, fasta_file):
    if os.path.exists(fasta_file):
        return
    with open(fasta_file, "w") as output_handle:
        for feature in record.features:
            if feature.type == "CDS":
                try:
                    # Get the protein ID or locus tag for the sequence ID in the FASTA file
                    protein_id = feature.qualifiers.get(
                        "protein_id", feature.qualifiers.get("locus_tag", ["Unknown"])
                    )[0]
                    # Try to get the translation from qualifiers
                    translation = feature.qualifiers.get("translation")[0]
                    if translation:
                        sequence = Seq(translation)
                    else:
                        # If translation not found, extract and translate the sequence
                        sequence = feature.location.extract(record).seq
                        sequence = sequence[
                            : len(sequence) - len(sequence) % 3
                        ].translate()
                    if len(sequence) > 800 or len(sequence) < 200:
                        continue
                    # Create a new SeqRecord and write to the output handle
                    SeqIO.write(
                        SeqIO.SeqRecord(sequence, id=protein_id, description=""),
                        output_handle,
                        "fasta",
                    )
                except Exception as e:
                    print(
                        "Error processing feature:",
                        e,
                        "in record",
                        record.id,
                        "for feature",
                        feature,
                    )
                    continue


def save_enzymes_to_fasta(record_dict):
    # Save enzymes together with reference to fasta
    for enzyme, results in record_dict.items():
        # Create a SeqRecord for the reference sequence
        reference_record = SeqRecord(
            Seq(enzymes[enzyme]["reference_for_alignment"]),
            id="Reference",
            description="Reference Sequence",
        )

        # Generate a list of SeqRecord objects from the results, with the reference sequence at the beginning
        seq_records = [reference_record] + [
            SeqRecord(Seq(result["sequence"]), id=id, description=result["product"])
            for id, result in results.items()
        ]

        fasta_name = os.path.join(tmp_dir, f"{enzyme}_tailoring_enzymes.fasta")
        SeqIO.write(seq_records, fasta_name, "fasta")


def classifier_prediction(feature_matrix, classifier_path, mode):
    """Predict values using a classifier"""
    label_encoder = LabelEncoder()
    if mode == "metabolism":
        label_encoder.classes_ = np.array(
            ["primary_metabolism", "secondary_metabolism"]
        )
        unique_count_target = 2
    elif mode == "BGC":
        label_encoder.classes_ = np.array(BGC_types)
        unique_count_target = len(BGC_types)
    else:
        raise ValueError(f"Mode {mode} not available")
    num_columns = feature_matrix.shape[1]
    logging.debug(f"Num columns: {num_columns}, classes {unique_count_target}")
    model_class = None
    if os.path.splitext(classifier_path)[1] == ".pth":
        # Check which model type is in the classifier path and assign the appropriate class
        if "_LSTM" in classifier_path:
            classifier = LSTM(
                in_features=num_columns,
                hidden_size=20,
                num_classes=unique_count_target,
            )
        elif "_BasicFFNN" in classifier_path:
            classifier = BasicFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_IntermediateFFNN" in classifier_path:
            classifier = IntermediateFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_AdvancedFFNN" in classifier_path:
            classifier = AdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_VeryAdvancedFFNN" in classifier_path:
            classifier = VeryAdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        else:
            raise ValueError("Unknown model type in the path")

        classifier.load_state_dict(torch.load(classifier_path))
        classifier.eval()
        classifier.to(device)
        with torch.no_grad():
            feature_matrix = torch.tensor(
                feature_matrix.to_numpy(), dtype=torch.float32
            ).to(device)
            logits = classifier(feature_matrix)

            predicted_values = torch.argmax(logits, dim=1).cpu().numpy()
            predicted_values = label_encoder.inverse_transform(predicted_values)
            score_predicted_values = F.softmax(logits, dim=1).cpu().numpy()
    else:
        with open(classifier_path, "rb") as file:
            classifier = pickle.load(file)

        predicted_values = classifier.predict(feature_matrix)
        score_predicted_values = classifier.predict_proba(feature_matrix)
    return predicted_values, score_predicted_values


def calculate_score(filtered_dataframe, target_BGC_type):
    score = 0
    hybrid_found = None
    primary_types = ["NRPS", "PKS"]
    secondary_types = ["Terpene", "Alkaloid"]
    relevant_types = []
    if target_BGC_type in primary_types:
        relevant_types = primary_types
    elif target_BGC_type in secondary_types:
        relevant_types = secondary_types
    for protein_id, row in filtered_dataframe.iterrows():
        # Calculate the adjusted score based on the specific details of the protein
        adjusted_score = (row["BGC_type_score"] + 0.7) * row["NP_BGC_affiliation_score"]
        logging.debug(
            f"Score = {score} score per protein = {adjusted_score}, protein = {protein_id})"
        )
        if row["BGC_type"] == target_BGC_type:  # Increase score for matching BGC type
            score += adjusted_score
        elif row["BGC_type"] in relevant_types:
            score += adjusted_score / 2
        else:  # Decrease score for non-matching BGC type
            score -= adjusted_score

        # Check for hybrid BGC types
        if (
            row["BGC_type"] in primary_types
            and target_BGC_type in primary_types
            and row["BGC_type"] != target_BGC_type
        ):
            hybrid_found = (
                "NRPS/PKS-hybrid" if target_BGC_type == "NRPS" else "PKS/NRPS-hybrid"
            )
        elif (
            row["BGC_type"] in secondary_types
            and target_BGC_type in secondary_types
            and row["BGC_type"] != target_BGC_type
        ):
            hybrid_found = (
                "Terpene/Alkaloid-hybrid"
                if target_BGC_type == "Terpene"
                else "Alkaloid/Terpene-hybrid"
            )

    return round(score, 3), hybrid_found if hybrid_found else target_BGC_type


def overlap_percentage(start1, end1, start2, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)
    length1 = end1 - start1
    length2 = end2 - start2
    return (overlap_length / min(length1, length2)) * 100


def process_dataframe_and_save(
    complete_dataframe, gb_record, output_base_path, score_threshold=0
):
    scores_list = []  # to store scores for histogram plotting
    results_dict = {}

    # Ensure output path has the proper trailing slash
    if not output_base_path.endswith("/"):
        output_base_path += "/"
    raw_BGCs = []
    for index, row in complete_dataframe.iterrows():
        window_start, window_end = adjust_window_size(
            row, complete_dataframe, len(gb_record.seq)
        )

        # Filter dataframe based on window
        filtered_dataframe = complete_dataframe[
            (complete_dataframe["cds_start"] >= window_start)
            & (complete_dataframe["cds_end"] <= window_end)
        ]

        # Compute score for filtered dataframe using the separate function
        score, BGC_type = calculate_score(filtered_dataframe, row["BGC_type"])
        scores_list.append(score)

        # Create features for each protein with its score
        for protein_id, row in filtered_dataframe.iterrows():
            feature = SeqFeature(
                FeatureLocation(start=row["cds_start"], end=row["cds_end"]),
                type="misc_feature",
                qualifiers={
                    "protein_id": protein_id,
                    "BGC_type_score": row["BGC_type_score"],
                    "NP_BGC_affiliation_score": row["NP_BGC_affiliation_score"],
                    "BGC_type": row["BGC_type"],
                    "metabolism_type": row["NP_BGC_affiliation"],
                    "note": f"Predicted using Tailenza 1.0.0",
                },
            )
            gb_record.features.append(feature)
    # Filter overlapping BGCs

    if score >= score_threshold:
        # Add a new feature to the GenBank record for this window if score is above threshold
        feature = SeqFeature(
            FeatureLocation(
                start=max(0, window_start),
                end=min(window_end, len(gb_record.seq)),
            ),
            type="misc_feature",
            qualifiers={
                "label": f"Score: {score} {BGC_type}",
                "note": f"Predicted using Tailenza 1.0.0",
            },
        )
        raw_BGCs.append(
            {
                "feature": BGC_record,
                "score": score,
                "begin": max(0, window_start),
                "end": min(window_end, len(gb_record.seq)),
            }
        )
    raw_BGCs.sort(key=lambda x: x["score"], reverse=True)  # Sort by score descending
    filtered_BGCs = []
    for annotation in raw_BGCs:
        overlap = False
        for fa in filtered_BGCs:
            overlap_percent = overlap_percentage(
                annotation["begin"],
                annotation["end"],
                fa["begin"],
                fa["end"],
            )
            if overlap_percent > 50:
                overlap = True
                if annotation["score"] > fa["score"]:
                    fa.update(annotation)
                break
        if not overlap:
            filtered_BGCs.append(annotation)
    # Setup the output path based on BGC type
    output_path = os.path.join(output_base_path + row["BGC_type"])
    os.makedirs(output_path, exist_ok=True)

    for feature, score, window_start, window_end in filtered_BGCs:

        gb_record.features.append(feature)
        BGC_record = gb_record[
            max(0, window_start) : min(window_end, len(gb_record.seq))
        ]
        BGC_record.annotations["molecule_type"] = "dna"
        filename_record = f"{gb_record.id}_{window_start}_{window_end}_{score}.gb"

        SeqIO.write(BGC_record, os.path.join(output_path, filename_record), "gb")

        results_dict[f"{gb_record.id}_{window_start}"] = {
            "ID": gb_record.id,
            "description": gb_record.description,
            "window_start": max(0, window_start),
            "window_end": min(window_end, len(gb_record.seq)),
            "score": score,
            "filename": filename_record,
        }
    SeqIO.write(
        gb_record,
        os.path.join(output_base_path, f"{gb_record.id}_tailenza_output.gb"),
        "gb",
    )

    return results_dict


def plot_histogram(scores):
    plt.hist(scores, bins=30, edgecolor="black")
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.output[0], "histogram_of_scores.png"), dpi=300)
    plt.grid(True)


def clear_tmp_dir(tmp_dir):
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def safe_map(row, mapping_dict, column_name):
    key = row.name  # row.name will give the index of the row
    if key in mapping_dict:
        row[column_name] = mapping_dict[key]
    return row


def adjust_window_size(row, complete_dataframe, len_record):
    primary_types = ["NRPS", "PKS"]
    secondary_types = ["Terpene", "Alkaloid"]

    if row["BGC_type"] in primary_types:
        initial_window = 60000
        trailing_window = 15000
        relevant_types = primary_types
    elif row["BGC_type"] in secondary_types:
        initial_window = 15000
        trailing_window = 5000
        relevant_types = secondary_types
    else:
        initial_window = 15000
        trailing_window = 5000
        relevant_types = [row["BGC_type"]]

    window_start = row["cds_start"]
    window_end = row["cds_end"]
    logging.debug(f"CDS: {window_start}, {window_end}")
    cds_starts = [window_start]

    # Extend window if additional TEs are found within the same BGC type or relevant related types and metabolism is secondary
    if row["NP_BGC_affiliation"] == "secondary_metabolism":
        while True:
            logging.debug(f"CDS: {window_start}, {window_end}")

            extended_window_end = window_end + initial_window
            extended_dataframe = complete_dataframe[
                (complete_dataframe["cds_start"] >= window_start)
                & (complete_dataframe["cds_end"] <= extended_window_end)
            ]
            logging.debug(f"extended dataframe {extended_dataframe}")
            new_tes = extended_dataframe[
                (
                    (extended_dataframe["BGC_type"] == row["BGC_type"])
                    | (extended_dataframe["BGC_type"].isin(relevant_types))
                )
                # & (extended_dataframe["NP_BGC_affiliation"] == "secondary_metabolism")
                & (~extended_dataframe["cds_start"].isin(cds_starts))
            ]
            cds_starts.extend(new_tes["cds_start"].to_list())
            logging.debug(f" TEs: {new_tes}")
            if new_tes.empty:
                break
            logging.debug(new_tes["cds_end"].max())
            logging.debug(window_end)
            window_end = max(window_end, int(new_tes["cds_end"].max()))

    # If only one TE is found, center the window around the TE
    if (window_end - window_start) == (row["cds_end"] - row["cds_start"]):
        logging.debug("only one CDS")
        window_center = (row["cds_start"] + row["cds_end"]) // 2
        logging.debug(window_center)
        window_start = max(0, window_center - initial_window // 2)
        window_end = window_center + (initial_window // 2)
    else:
        window_end = max(window_start + initial_window, window_end)
    return max(0, window_start - trailing_window), min(
        window_end + trailing_window, len_record
    )


def main():

    args = parser.parse_args()
    filename = args.input[0]
    package_dir = files("tailenza").joinpath("")
    score_threshold = args.score_cutoff

    try:
        os.mkdir(args.output)
    except:
        logging.info("WARNING: output directory already existing and not empty.")

    # Temporary directory for intermediate files
    global tmp_dir
    tmp_dir = os.path.join(args.output, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    if not (
        filename.endswith(".gbff")
        or filename.endswith(".gb")
        or filename.endswith(".gbk")
    ):
        raise ValueError("Input file must be a GenBank file.")
    # Load the ESM-1b model

    file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(file_path_model)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    logging.debug("Model type: %s", type(model))
    logging.debug("Alphabet type: %s", type(alphabet))
    logging.debug("Batch converter type: %s", type(batch_converter))
    logging.debug("Model loaded successfully")
    for gb_record in SeqIO.parse(filename, "genbank"):
        # Create datastructure for results and fill with hmmer results
        tailoring_enzymes_in_record = {
            key: run_hmmer(gb_record, key) for key in enzymes
        }
        enzyme_dataframes = {
            enzyme_name: set_dataframe_columns(
                process_feature_dict(enzyme_dict, enzyme_name)
            )
            for enzyme_name, enzyme_dict in tailoring_enzymes_in_record.items()
        }
        complete_dataframe = pd.concat(
            [enzyme_dataframe for enzyme_dataframe in enzyme_dataframes.values()],
            axis=0,
        )
        # Save enzymes together with reference to fasta for running the alignment on it
        save_enzymes_to_fasta(tailoring_enzymes_in_record)
        fasta_dict = {
            key: os.path.join(tmp_dir, f"{key}_tailoring_enzymes.fasta")
            for key in enzymes
        }
        alignments = {
            enzyme: muscle_align_sequences(filename, enzyme)
            for enzyme, filename in fasta_dict.items()
        }
        logging.debug(f"Aligments {alignments}")
        logging.debug(f"First alignment {alignments['P450']}")
        fragment_matrixes = {
            key: (
                fragment_alignment(
                    alignments[key],
                    enzymes[key]["splitting_list"],
                    fastas_aligned_before,
                )
                if len(alignments[key]) > 1
                else None
            )
            for key in enzymes
        }
        logging.debug(
            f"fragment_matrices: {[fragment_matrix.head() for fragment_matrix in fragment_matrixes.values() if fragment_matrix is not None]}"
        )
        feature_matrixes = {
            key: (
                featurize_fragments(
                    fragment_matrix,
                    batch_converter,
                    model,
                    include_charge_features,
                    device,
                )
                if fragment_matrix is not None
                else pd.DataFrame()
            )
            for key, fragment_matrix in fragment_matrixes.items()
        }
        logging.debug(
            f"feature_matrices: {[feature_matrix.head() for feature_matrix in feature_matrixes.values() if feature_matrix is not None]}"
        )

        classifiers_metabolism_paths = {
            key: directory_of_classifiers_NP_affiliation
            + key
            + enzymes[key]["classifier_metabolism"]
            for key in enzymes
        }
        classifiers_BGC_type_paths = {
            key: directory_of_classifiers_BGC_type
            + key
            + enzymes[key]["classifier_BGC_type"]
            for key in enzymes
        }
        predicted_BGC_types = {}
        scores_predicted_BGC_type = {}
        predicted_metabolism_types = {}
        scores_predicted_metabolism = {}
        for key, feature_matrix in feature_matrixes.items():
            logging.debug(f"feature matrix {feature_matrix.head()}, key {key}")
            if key == "ycao":
                if not feature_matrix.empty:
                    predicted_BGC_types[key] = ["RiPPs"] * len(feature_matrix)
                    scores_predicted_BGC_type[key] = [[1]] * len(feature_matrix)
                    predicted_values_metabolism, score_predicted_values_metabolism = (
                        classifier_prediction(
                            feature_matrix,
                            classifiers_metabolism_paths[key],
                            "metabolism",
                        )
                    )
                    predicted_metabolism_types[key] = predicted_values_metabolism
                    scores_predicted_metabolism[key] = score_predicted_values_metabolism
                else:
                    predicted_BGC_types[key] = []
                    scores_predicted_BGC_type[key] = []
                    predicted_metabolism_types[key] = []
                    scores_predicted_metabolism[key] = []

            else:
                if not feature_matrix.empty:
                    predicted_values_BGC, score_predicted_values_BGC = (
                        classifier_prediction(
                            feature_matrix, classifiers_BGC_type_paths[key], "BGC"
                        )
                    )
                    predicted_BGC_types[key] = predicted_values_BGC
                    scores_predicted_BGC_type[key] = score_predicted_values_BGC

                    predicted_values_metabolism, score_predicted_values_metabolism = (
                        classifier_prediction(
                            feature_matrix,
                            classifiers_metabolism_paths[key],
                            "metabolism",
                        )
                    )
                    predicted_metabolism_types[key] = predicted_values_metabolism
                    scores_predicted_metabolism[key] = score_predicted_values_metabolism
                else:
                    predicted_BGC_types[key] = []
                    scores_predicted_BGC_type[key] = []
                    predicted_metabolism_types[key] = []
                    scores_predicted_metabolism[key] = []

        for enzyme in enzymes:

            # Create a dictionary mapping for each predicted value
            logging.debug(predicted_metabolism_types)
            logging.debug(enzyme_dataframes)
            predicted_metabolism_dict = dict(
                zip(enzyme_dataframes[enzyme].index, predicted_metabolism_types[enzyme])
            )
            score_predicted_metabolism_dict = dict(
                zip(
                    enzyme_dataframes[enzyme].index,
                    [
                        scores[1]
                        for prediction, scores in zip(
                            predicted_metabolism_types[enzyme],
                            scores_predicted_metabolism[enzyme],
                        )
                    ],
                )
            )

            predicted_BGC_type_dict = dict(
                zip(enzyme_dataframes[enzyme].index, predicted_BGC_types[enzyme])
            )
            score_predicted_BGC_type_dict = dict(
                zip(
                    enzyme_dataframes[enzyme].index,
                    [
                        max(scores)
                        for prediction, scores in zip(
                            predicted_BGC_types[enzyme],
                            scores_predicted_BGC_type[enzyme],
                        )
                    ],
                )
            )
            # Map the predictions and scores to the dataframe using the dictionaries

            # Apply the function for NP_BGC_affiliation and its score
            complete_dataframe = complete_dataframe.apply(
                safe_map,
                args=(predicted_metabolism_dict, "NP_BGC_affiliation"),
                axis=1,
            )
            complete_dataframe = complete_dataframe.apply(
                safe_map,
                args=(score_predicted_metabolism_dict, "NP_BGC_affiliation_score"),
                axis=1,
            )

            # Apply the function for BGC_type and its score
            complete_dataframe = complete_dataframe.apply(
                safe_map, args=(predicted_BGC_type_dict, "BGC_type"), axis=1
            )
            complete_dataframe = complete_dataframe.apply(
                safe_map,
                args=(score_predicted_BGC_type_dict, "BGC_type_score"),
                axis=1,
            )
        complete_dataframe.to_csv(
            os.path.join(args.output, f"complete_dataframe_{gb_record.id}.csv")
        )
        results_dict = process_dataframe_and_save(
            complete_dataframe,
            gb_record,
            args.output,
            score_threshold=score_threshold,
        )
        result_df = pd.DataFrame(results_dict)
        result_df.to_csv(
            os.path.join(args.output, f"result_dataframe_{gb_record.id}.csv")
        )
    clear_tmp_dir(tmp_dir)


if __name__ == "__main__":
    main()
