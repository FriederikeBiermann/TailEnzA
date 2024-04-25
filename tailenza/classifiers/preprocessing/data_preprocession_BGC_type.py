import os
import Bio
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from feature_generation import fragment_alignment, featurize_fragments
from tailenza.data.enzyme_information import BGC_types, enzymes
from importlib.resources import files
from pathlib import Path
import torch
import esm
import importlib.resources as pkg_resources
import logging

DEBUGGING = True
# if fastas aligned before True-> use alignment made for instance in geneious utilizing MUSCLE align -> best for larger datasets
FASTAS_ALIGNED_BEFORE = True
# if include_charge_features=True-> features describing general electrostatic information will be included
INCLUDE_CHARGE_FEATURES = True

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


package_dir = files("tailenza").joinpath("")
logging.debug("Package directory: %s", package_dir)
foldername_training_sets = "training_data/Toy_dataset"
foldername_output = os.path.join(
    package_dir,
    "data",
    "classifiers",
    "Test_transformer",
)

# For debugging
if DEBUGGING:
    enzymes = {
        "ycao": {
            "splitting_list": {
                "begin": [0, 64],
                "sbr1": [65, 82],
                "f2": [83, 153],
                "sbr2": [154, 185],
                "f3": [186, 227],
                "sbr3": [228, 281],
                "f4": [282, 296],
                "sbr4": [297, 306],
                "f5": [307, 362],
                "sbr5": [363, 368],
                "end": [369, 350],
            },
            "hmm_file": "ycao.hmm",
            "classifier_BGC_type": "_ExtraTreesClassifier_classifier.sav",
            "classifier_metabolism": "_ExtraTreesClassifier_classifier.sav",
            "reference_for_alignment": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
            "gap_opening_penalty": -2,
            "gap_extend_penalty": -1,
        }
    }
if DEBUGGING:
    BGC_types = ["ripp"]


def create_filenames(enzyme, BGC_types, foldername_training_sets):
    """
    Create a dictionary of filenames categorized by BGC types for a specific enzyme.

    Parameters:
    - enzyme (str): The name of the enzyme to filter files by.
    - BGC_types (list of str): A list of biosynthetic gene cluster (BGC) types to categorize the files.
    - foldername_training_sets (str): The path to the directory containing the training set files.

    Returns:
    - dict: A dictionary where keys are BGC types, and values are lists of filenames that belong to each BGC type.
    """
    all_files = os.listdir(foldername_training_sets)
    filenames_dict = {BGC_type: [] for BGC_type in BGC_types}

    for file in all_files:
        for BGC_type in BGC_types:
            if f"{enzyme}_" in file and BGC_type in file:
                filenames_dict[BGC_type].append(
                    os.path.join(foldername_training_sets, file)
                )
    # For debugging
    if DEBUGGING:
        filenames_dict = {
            "RiPPs": [
                os.path.join(foldername_training_sets, "toy_ripp_YCAO alignment.fas")
            ]
        }

    return filenames_dict


def process_datasets(
    foldername_training_sets, model, batch_converter, include_charge_features=True
):
    """
    Process datasets to generate a feature matrix for each enzyme-BGC type pair.

    Parameters:
    - foldername_training_sets (str): The path to the directory containing the training set files.
    - model: The pre-trained transformer model used for generating sequence embeddings.
    - batch_converter: A utility function provided by the transformer model for converting sequences into a compatible format.
    - include_charge_features (bool): Flag to indicate whether to include electrostatic charge features in the feature matrix.

    Returns:
    - None: The function saves the complete feature matrix to a CSV file and prints the path to this file.
    """
    complete_feature_matrix = pd.DataFrame()

    for enzyme in enzymes:
        filenames_dict = create_filenames(enzyme, BGC_types, foldername_training_sets)
        logging.debug("Filenames dictionary created for %s", enzyme)
        logging.debug(filenames_dict)

        for BGC_type, datasets in filenames_dict.items():
            for dataset in datasets:
                logging.debug("Processing dataset: %s", dataset)
                msa_path = Path(dataset)
                logging.debug(enzymes)
                splitting_list = enzymes[enzyme]["splitting_list"]
                logging.debug("Splitting list created for %s", enzyme)
                logging.debug(splitting_list)
                alignment = AlignIO.read(msa_path, "fasta")
                fragment_matrix = fragment_alignment(
                    alignment, splitting_list, FASTAS_ALIGNED_BEFORE
                )
                logging.debug("Fragment matrix created for %s", enzyme)
                logging.debug(fragment_matrix)
                feature_matrix = featurize_fragments(
                    fragment_matrix, batch_converter, model, include_charge_features
                )
                logging.debug("Feature matrix created for %s", enzyme)
                feature_matrix["target"] = BGC_type
                complete_feature_matrix = pd.concat(
                    [complete_feature_matrix, feature_matrix], ignore_index=True
                )

    output_path = Path(foldername_output, "complete_feature_matrix.csv")
    complete_feature_matrix.to_csv(output_path, index=False)
    logging.info("Feature matrix saved to %s", output_path)


if __name__ == "__main__":
    # Load the ESM-1b model

    file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
    print(file_path_model)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(file_path_model)
    model = model.eval()
    batch_converter = alphabet.get_batch_converter()
    logging.debug("Model type: %s", type(model))
    logging.debug("Alphabet type: %s", type(alphabet))
    logging.debug("Batch converter type: %s", type(batch_converter))
    logging.debug("Model loaded successfully")

    process_datasets(
        foldername_training_sets, model, batch_converter, INCLUDE_CHARGE_FEATURES
    )
