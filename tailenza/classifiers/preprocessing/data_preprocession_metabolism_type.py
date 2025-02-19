import os
import pandas as pd
from Bio import AlignIO, SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import torch
import esm
import logging
from typing import List, Dict, Optional, Callable, Tuple
from tailenza.classifiers.preprocessing.feature_generation import AlignmentDataset
from tailenza.data.enzyme_information import BGC_types, enzymes
from importlib.resources import files

DEBUGGING = False
# if fastas aligned before True-> use alignment made for instance in geneious utilizing MUSCLE align -> best for larger datasets
FASTAS_ALIGNED_BEFORE = True
# if include_charge_features=True-> features describing general electrostatic information will be included
INCLUDE_CHARGE_FEATURES = True

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)  # run on second GPU
package_dir = files("tailenza").joinpath("")
logging.debug("Package directory: %s", package_dir)
foldername_training_sets = "training_data/ncbi_dataset_16_05_2024_without_divergent"
foldername_output = "preprocessed_data/dataset_transformer_new"

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
                "end": [369, 369],
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


def create_filenames(
    enzyme: str, BGC_types: List[str], foldername_training_sets: str
) -> Dict[str, List[str]]:
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

    # Initialize a dictionary to store filenames for each BGC type
    filenames_dict: Dict[str, List[str]] = {BGC_type: [] for BGC_type in BGC_types}
    filenames_dict["NCBI"] = []

    # Populate the dictionary based on the enzyme and BGC types
    for file in all_files:
        if f"{enzyme}_" in file and "NCBI" in file:
            filenames_dict["NCBI"].append(os.path.join(foldername_training_sets, file))
        else:
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
    foldername_training_sets: str,
    model: torch.nn.Module,
    batch_converter: Callable[
        [List[Tuple[str, str]]], Tuple[List[str], torch.Tensor, torch.Tensor]
    ],
    include_charge_features: bool = True,
) -> None:
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
    model.to(device)
    for enzyme in enzymes:
        filenames_dict = create_filenames(enzyme, BGC_types, foldername_training_sets)
        logging.debug("Filenames dictionary created for %s", enzyme)
        logging.debug(filenames_dict)

        for BGC_type, datasets in filenames_dict.items():
            for dataset in datasets:
                logging.debug("Processing dataset: %s", dataset)
                msa_path = Path(dataset)
                logging.debug(enzymes)

                # Use the AlignmentDataset class to handle dataset processing
                alignment = AlignIO.read(msa_path, "fasta")
                dataset_obj = AlignmentDataset(enzymes, enzyme, alignment)

                # Filter alignment based on sequence length
                filtered_alignment = dataset_obj._filter_alignment()
                dataset_obj.alignment = filtered_alignment

                # Perform fragment alignment
                fragment_matrix = dataset_obj.fragment_alignment(
                    fastas_aligned_before=FASTAS_ALIGNED_BEFORE
                )
                logging.debug("Fragment matrix created for %s", enzyme)
                logging.debug(fragment_matrix)

                # Featurize fragments
                feature_matrix = dataset_obj.featurize_fragments(
                    batch_converter, model, device=device
                )
                logging.debug("Feature matrix created for %s", enzyme)
                if BGC_type in BGC_types:
                    metabolism_type = "secondary_metabolism"
                elif BGC_type == "NCBI":
                    metabolism_type = "primary_metabolism"
                else:
                    metabolism_type = "unknown"
                feature_matrix["target"] = metabolism_type
                complete_feature_matrix = pd.concat(
                    [complete_feature_matrix, feature_matrix], ignore_index=True
                )

        output_path = Path(
            foldername_output, f"{enzyme}_metabolism_type_feature_matrix.csv"
        )
        complete_feature_matrix.to_csv(output_path, index=False)
        logging.info("Feature matrix saved to %s", output_path)


if __name__ == "__main__":
    # Load the ESM-1b model

    file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
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
