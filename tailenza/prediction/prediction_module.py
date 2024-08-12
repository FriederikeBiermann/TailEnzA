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
from tailenza.classifiers.preprocessing.feature_generation import AlignmentDataset

import logging
from importlib.resources import files
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch import nn
from typing import List, Dict, Callable, Tuple, Optional

from tailenza.classifiers.machine_learning.machine_learning_training_classifiers_AA_BGC_type import (
    LSTM,
    BasicFFNN,
    IntermediateFFNN,
    AdvancedFFNN,
    VeryAdvancedFFNN,
)

# Suppress warnings
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

# Argument parser setup
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
    default=["Output/"],
    help="Output directory",
)

parser.add_argument(
    "-c",
    "--score_cutoff",
    type=float,
    nargs=1,
    metavar="cutoff",
    default=[0.75],
    help="Cutoff score to use for the genbank extraction.",
)

parser.add_argument(
    "-d",
    "--device",
    type=str,
    nargs=1,
    metavar="device",
    default=["cuda:0"],
    help="Cuda device to run extraction on.",
)

args = parser.parse_args()
directory_of_classifiers_BGC_type = "../classifiers/classifiers/BGC_type/"
directory_of_classifiers_NP_affiliation = "../classifiers/classifiers/metabolism_type/"
fastas_aligned_before = True
include_charge_features = True
package_dir = files("tailenza").joinpath("")
hmm_dir = package_dir.joinpath("data", "HMM_files")
device = torch.device(args.device[0] if torch.cuda.is_available() else "cpu")


def extract_feature_properties(feature: SeqFeature) -> Dict[str, str]:
    """Extracts properties from a GenBank CDS feature.

    Args:
        feature (SeqFeature): The feature from which to extract properties.

    Returns:
        Dict[str, str]: A dictionary containing the sequence, product, cds_start, and cds_end.
    """
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


def get_identifier(feature: SeqFeature) -> str:
    """Returns the 'locus_tag' or 'protein_id' from a feature.

    Args:
        feature (SeqFeature): The feature from which to extract the identifier.

    Returns:
        str: The identifier of the feature.
    """
    return feature.qualifiers.get(
        "protein_id", feature.qualifiers.get("locus_tag", ["Unknown"])
    )[0]


def process_feature_dict(
    product_dict: Dict[str, Dict[str, str]], enzyme_name: str
) -> pd.DataFrame:
    """Process the feature dictionary and return a DataFrame.

    Args:
        product_dict (Dict[str, Dict[str, str]]): The dictionary containing feature data.
        enzyme_name (str): The name of the enzyme.

    Returns:
        pd.DataFrame: A DataFrame with processed feature data.
    """
    if product_dict:
        df = pd.DataFrame(product_dict).transpose()
        df.insert(0, "Enzyme", enzyme_name)
    else:
        df = pd.DataFrame(columns=["sequence", "product", "cds_start", "cds_end"])
        df.insert(0, "Enzyme", enzyme_name)
    return df


def set_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sets default columns to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The modified DataFrame with additional columns.
    """
    df["BGC_type"] = ""
    df["BGC_type_score"] = ""
    df["NP_BGC_affiliation"] = ""
    df["NP_BGC_affiliation_score"] = ""
    return df


def muscle_align_sequences(
    fasta_filename: str, enzyme: str
) -> AlignIO.MultipleSeqAlignment:
    """Align sequences using MUSCLE and return the alignment.

    Args:
        fasta_filename (str): The name of the FASTA file containing sequences.
        enzyme (str): The name of the enzyme.

    Returns:
        AlignIO.MultipleSeqAlignment: The aligned sequences.
    """
    num_sequences = sum(1 for _ in SeqIO.parse(fasta_filename, "fasta"))
    if num_sequences <= 1:
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
        "-center",
        str(enzymes[enzyme]["center"]),
    ]

    try:
        subprocess.check_call(muscle_cmd, stdout=DEVNULL, stderr=DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Error: Failed to run command {' '.join(muscle_cmd)}")
        sys.exit(1)

    return AlignIO.read(open(f"{fasta_filename}_aligned.fasta"), "fasta")


def create_feature_lookup(record: SeqRecord) -> Dict[str, SeqFeature]:
    """Create a lookup dictionary from the record's features.

    Args:
        record (SeqRecord): The GenBank record to extract features from.

    Returns:
        Dict[str, SeqFeature]: A dictionary keyed by protein_id or locus_tag.
    """
    feature_lookup = {}
    for feature in record.features:
        if feature.type == "CDS":
            protein_id = feature.qualifiers.get(
                "protein_id", feature.qualifiers.get("locus_tag", ["Unknown"])
            )[0]
            feature_lookup[protein_id] = feature
    return feature_lookup


def run_hmmer(record: SeqRecord, enzyme: str) -> Dict[str, Dict[str, str]]:
    """Run HMMER to search for enzyme-specific hits in the sequence.

    Args:
        record (SeqRecord): The GenBank record containing the sequence.
        enzyme (str): The enzyme being searched for.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary of features that matched the HMM profile.
    """
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
                        if evalue >= 10e-15:
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


def genbank_to_fasta_cds(record: SeqRecord, fasta_file: str) -> None:
    """Convert GenBank CDS features to a FASTA file.

    Args:
        record (SeqRecord): The GenBank record.
        fasta_file (str): The output FASTA file.
    """
    if os.path.exists(fasta_file):
        return
    with open(fasta_file, "w") as output_handle:
        for feature in record.features:
            if feature.type == "CDS":
                try:
                    protein_id = feature.qualifiers.get(
                        "protein_id", feature.qualifiers.get("locus_tag", ["Unknown"])
                    )[0]
                    translation = feature.qualifiers.get("translation")[0]
                    sequence = (
                        Seq(translation)
                        if translation
                        else feature.location.extract(record).seq.translate()
                    )
                    if 200 <= len(sequence) <= 800:
                        SeqIO.write(
                            SeqRecord(sequence, id=protein_id, description=""),
                            output_handle,
                            "fasta",
                        )
                except Exception as e:
                    logging.error(
                        f"Error processing feature {feature} in record {record.id}: {e}"
                    )
                    continue


def save_enzymes_to_fasta(record_dict: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    """Save enzymes together with the reference to a FASTA file.

    Args:
        record_dict (Dict[str, Dict[str, Dict[str, str]]]): A dictionary of enzymes and their features.
    """
    for enzyme, results in record_dict.items():
        reference_record = SeqRecord(
            Seq(enzymes[enzyme]["reference_for_alignment"]),
            id="Reference",
            description="Reference Sequence",
        )
        seq_records = [reference_record] + [
            SeqRecord(Seq(result["sequence"]), id=id, description=result["product"])
            for id, result in results.items()
        ]
        fasta_name = os.path.join(tmp_dir, f"{enzyme}_tailoring_enzymes.fasta")
        SeqIO.write(seq_records, fasta_name, "fasta")


def classifier_prediction(
    feature_matrix: pd.DataFrame, classifier_path: str, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict values using a classifier.

    Args:
        feature_matrix (pd.DataFrame): The feature matrix to use for predictions.
        classifier_path (str): The path to the classifier model.
        mode (str): The prediction mode ('metabolism' or 'BGC').

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted values and their associated scores.
    """
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
        classifier = get_classifier_by_path(
            classifier_path, num_columns, unique_count_target
        )
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


def get_classifier_by_path(
    classifier_path: str, num_columns: int, unique_count_target: int
) -> nn.Module:
    """Determine and return the appropriate classifier based on the file path.

    Args:
        classifier_path (str): The path to the classifier.
        num_columns (int): The number of columns in the feature matrix.
        unique_count_target (int): The number of unique classes.

    Returns:
        nn.Module: The classifier model.
    """
    if "_LSTM" in classifier_path:
        return LSTM(
            in_features=num_columns, hidden_size=20, num_classes=unique_count_target
        )
    elif "_BasicFFNN" in classifier_path:
        return BasicFFNN(num_classes=unique_count_target, in_features=num_columns)
    elif "_IntermediateFFNN" in classifier_path:
        return IntermediateFFNN(
            num_classes=unique_count_target, in_features=num_columns
        )
    elif "_AdvancedFFNN" in classifier_path:
        return AdvancedFFNN(num_classes=unique_count_target, in_features=num_columns)
    elif "_VeryAdvancedFFNN" in classifier_path:
        return VeryAdvancedFFNN(
            num_classes=unique_count_target, in_features=num_columns
        )
    else:
        raise ValueError("Unknown model type in the path")


def calculate_score(
    filtered_dataframe: pd.DataFrame, target_BGC_type: str
) -> Tuple[float, str]:
    """Calculate the score for a given dataframe filtered by a BGC type.

    Args:
        filtered_dataframe (pd.DataFrame): The filtered dataframe to score.
        target_BGC_type (str): The target BGC type.

    Returns:
        Tuple[float, str]: The calculated score and hybrid BGC type if applicable.
    """
    logging.debug(f"Filtered dataframe {filtered_dataframe}")
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
        logging.debug(row)
        adjusted_score = (row["BGC_type_score"] + 0.7) * row["NP_BGC_affiliation_score"]
        logging.debug(
            f"Score = {score} score per protein = {adjusted_score}, protein = {protein_id})"
        )
        if row["BGC_type"] == target_BGC_type:
            score += adjusted_score
        elif row["BGC_type"] in relevant_types:
            score += adjusted_score / 2
        else:
            score -= adjusted_score

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


def overlap_percentage(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculate the percentage of overlap between two genomic regions.

    Args:
        start1 (int): Start position of the first region.
        end1 (int): End position of the first region.
        start2 (int): Start position of the second region.
        end2 (int): End position of the second region.

    Returns:
        float: The percentage overlap between the two regions.
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)
    length1 = end1 - start1
    length2 = end2 - start2
    return (overlap_length / min(length1, length2)) * 100


def process_dataframe_and_save(
    complete_dataframe: pd.DataFrame,
    gb_record: SeqRecord,
    output_base_path: str,
    score_threshold: float = 0,
) -> Dict[str, Dict[str, str]]:
    """Process the dataframe, filter based on score, and save results.

    Args:
        complete_dataframe (pd.DataFrame): The complete dataframe with all features.
        gb_record (SeqRecord): The GenBank record being processed.
        output_base_path (str): The base path for saving output files.
        score_threshold (float, optional): The minimum score threshold for saving. Defaults to 0.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary of results including BGC information and scores.
    """
    scores_list = []
    results_dict = {}
    logging.debug(complete_dataframe)
    if not output_base_path.endswith("/"):
        output_base_path += "/"
    raw_BGCs = []
    for index, row in complete_dataframe.iterrows():
        window_start, window_end = adjust_window_size(
            row, complete_dataframe, len(gb_record.seq)
        )
        filtered_dataframe = complete_dataframe[
            (complete_dataframe["cds_start"] >= window_start)
            & (complete_dataframe["cds_end"] <= window_end)
        ]
        score, BGC_type = calculate_score(filtered_dataframe, row["BGC_type"])
        if BGC_type in ["NRPS", "PKS"]:
            score = (score / max(60_000, window_end - window_start)) * 60_000
        else:
            score = (score / max(15_000, window_end - window_start)) * 15_000

        scores_list.append(score)
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
        if score >= score_threshold:
            feature = SeqFeature(
                FeatureLocation(
                    start=max(0, int(window_start)),
                    end=min(int(window_end), len(gb_record.seq)),
                ),
                type="misc_feature",
                qualifiers={
                    "label": f"Score: {score} {BGC_type}",
                    "note": f"Predicted using Tailenza 1.0.0",
                },
            )
            raw_BGCs.append(
                {
                    "feature": feature,
                    "score": score,
                    "begin": max(0, int(window_start)),
                    "end": min(int(window_end), len(gb_record.seq)),
                    "BGC_type": row["BGC_type"],
                }
            )
    raw_BGCs.sort(key=lambda x: x["score"], reverse=True)
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
    logging.debug(filtered_BGCs)
    for feature, score, window_start, window_end, BGC_type in [
        BGC.values() for BGC in filtered_BGCs
    ]:
        output_path = os.path.join(output_base_path + BGC_type)
        os.makedirs(output_path, exist_ok=True)

        gb_record.features.append(feature)
        BGC_record = gb_record[
            max(0, int(window_start)) : min(int(window_end), len(gb_record.seq))
        ]
        BGC_record.annotations["molecule_type"] = "dna"
        filename_record = f"{gb_record.id}_{window_start}_{window_end}_{score}.gb"

        SeqIO.write(BGC_record, os.path.join(output_path, filename_record), "gb")

        results_dict[f"{gb_record.id}_{window_start}"] = {
            "ID": gb_record.id,
            "description": gb_record.description,
            "window_start": max(0, int(window_start)),
            "window_end": min(int(window_end), len(gb_record.seq)),
            "BGC_type": BGC_type,
            "score": score,
            "filename": filename_record,
        }
    SeqIO.write(
        gb_record,
        os.path.join(output_base_path, f"{gb_record.id}_tailenza_output.gb"),
        "gb",
    )

    return results_dict


def plot_histogram(scores: List[float]) -> None:
    """Plot and save a histogram of scores.

    Args:
        scores (List[float]): List of scores to plot.
    """
    plt.hist(scores, bins=30, edgecolor="black")
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.output[0], "histogram_of_scores.png"), dpi=300)
    plt.grid(True)


def clear_tmp_dir(tmp_dir: str) -> None:
    """Clear the temporary directory.

    Args:
        tmp_dir (str): Path to the temporary directory.
    """
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def safe_map(
    row: pd.Series, mapping_dict: Dict[str, str], column_name: str
) -> pd.Series:
    """Safely map values from a dictionary to a DataFrame column.

    Args:
        row (pd.Series): The row to map.
        mapping_dict (Dict[str, str]): The mapping dictionary.
        column_name (str): The name of the column to update.

    Returns:
        pd.Series: The updated row.
    """
    key = row.name
    if key in mapping_dict:
        row[column_name] = mapping_dict[key]
    return row


def adjust_window_size(
    row: pd.Series, complete_dataframe: pd.DataFrame, len_record: int
) -> Tuple[int, int]:
    """Adjust the window size around a feature based on its BGC type.

    Args:
        row (pd.Series): The row of the dataframe representing the feature.
        complete_dataframe (pd.DataFrame): The complete dataframe.
        len_record (int): The length of the sequence record.

    Returns:
        Tuple[int, int]: The start and end of the adjusted window.
    """
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
                & (~extended_dataframe["cds_start"].isin(cds_starts))
            ]
            cds_starts.extend(new_tes["cds_start"].to_list())
            logging.debug(f" TEs: {new_tes}")
            if new_tes.empty:
                break
            logging.debug(new_tes["cds_end"].max())
            logging.debug(window_end)
            window_end = max(window_end, int(new_tes["cds_end"].max()))

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


def main() -> None:
    """Main function for processing GenBank files and predicting BGC types."""
    args = parser.parse_args()
    filename = args.input[0]
    package_dir = files("tailenza").joinpath("")
    score_threshold = args.score_cutoff[0]

    try:
        os.mkdir(args.output[0])
    except FileExistsError:
        logging.info("WARNING: output directory already existing and not empty.")

    global tmp_dir
    tmp_dir = os.path.join(args.output[0], "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    if not filename.endswith((".gbff", ".gb", ".gbk")):
        raise ValueError("Input file must be a GenBank file.")

    file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(file_path_model)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    logging.debug("Model type: %s", type(model))
    logging.debug("Alphabet type: %s", type(alphabet))
    logging.debug("Batch converter type: %s", type(batch_converter))
    logging.debug("Model loaded successfully")

    for gb_record in SeqIO.parse(filename, "genbank"):
        tailoring_enzymes_in_record = {
            key: run_hmmer(gb_record, key) for key in enzymes
        }
        enzyme_dataframes = {
            enzyme_name: set_dataframe_columns(
                process_feature_dict(enzyme_dict, enzyme_name)
            )
            for enzyme_name, enzyme_dict in tailoring_enzymes_in_record.items()
        }

        save_enzymes_to_fasta(tailoring_enzymes_in_record)

        alignments = {
            enzyme: muscle_align_sequences(
                os.path.join(tmp_dir, f"{enzyme}_tailoring_enzymes.fasta"), enzyme
            )
            for enzyme in enzymes
        }

        logging.debug(f"Alignments: {alignments}")
        logging.debug(f"First alignment: {alignments['P450']}")

        # Using AlignmentDataset for processing
        feature_matrixes = {}
        for enzyme, alignment in alignments.items():
            if len(alignment) > 1:
                dataset = AlignmentDataset(enzymes, enzyme, alignment)
                dataset.filter_alignment()
                dataset.fragment_alignment(fastas_aligned_before=True)
                feature_matrix = dataset.featurize_fragments(
                    batch_converter, model, device
                )
                feature_matrixes[enzyme] = feature_matrix
            else:
                feature_matrixes[enzyme] = pd.DataFrame()

        logging.debug(
            f"Feature matrices: {[feature_matrix.head() for feature_matrix in feature_matrixes.values() if not feature_matrix.empty]}"
        )

        enzyme_dataframes_filtered = {}
        for enzyme, df in enzyme_dataframes.items():
            if len(df) != len(feature_matrixes[enzyme]):
                df_filtered = df[df.index.isin(feature_matrixes[enzyme].index)]
            else:
                df_filtered = df
            enzyme_dataframes_filtered[enzyme] = df_filtered
        enzyme_dataframes = enzyme_dataframes_filtered

        complete_dataframe = pd.concat(
            [enzyme_dataframe for enzyme_dataframe in enzyme_dataframes.values()],
            axis=0,
        )

        classifiers_metabolism_paths = {
            key: os.path.join(
                directory_of_classifiers_NP_affiliation,
                key + enzymes[key]["classifier_metabolism"],
            )
            for key in enzymes
        }
        classifiers_BGC_type_paths = {
            key: os.path.join(
                directory_of_classifiers_BGC_type,
                key + enzymes[key]["classifier_BGC_type"],
            )
            for key in enzymes
        }

        predicted_BGC_types = {}
        scores_predicted_BGC_type = {}
        predicted_metabolism_types = {}
        scores_predicted_metabolism = {}

        for key, feature_matrix in feature_matrixes.items():
            logging.debug(f"Feature matrix for {key}: {feature_matrix.head()}")

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
            logging.debug(f"Scores: {score_predicted_BGC_type_dict}")

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
            complete_dataframe = complete_dataframe.apply(
                safe_map, args=(predicted_BGC_type_dict, "BGC_type"), axis=1
            )
            complete_dataframe = complete_dataframe.apply(
                safe_map,
                args=(score_predicted_BGC_type_dict, "BGC_type_score"),
                axis=1,
            )

        complete_dataframe.to_csv(
            os.path.join(args.output[0], f"complete_dataframe_{gb_record.id}.csv")
        )

        results_dict = process_dataframe_and_save(
            complete_dataframe,
            gb_record,
            args.output[0],
            score_threshold=score_threshold,
        )

        result_df = pd.DataFrame(results_dict)
        result_df.to_csv(
            os.path.join(args.output[0], f"result_dataframe_{gb_record.id}.csv")
        )

    clear_tmp_dir(tmp_dir)


if __name__ == "__main__":
    main()
