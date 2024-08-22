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
from typing import List, Dict, Callable, Tuple, Optional, Any

from tailenza.classifiers.machine_learning.machine_learning_training_classifiers_AA_BGC_type import (
    LSTM,
    BasicFFNN,
    IntermediateFFNN,
    AdvancedFFNN,
    VeryAdvancedFFNN,
)


class Predictor:
    def __init__(self, classifier_path: str, device: str):
        """
        Initialize the Prediction class.

        Args:
            classifier_path (str): The path to the classifier model.
            device (str): Computational device to use (e.g., "cpu", "cuda").
        """
        self.classifier_path = classifier_path
        self.device = device
        self.classifier = None
        self.label_encoder = LabelEncoder()

    def load_classifier(self, num_columns: int, unique_count_target: int):
        """
        Load the appropriate classifier based on the file path.

        Args:
            num_columns (int): The number of columns in the feature matrix.
            unique_count_target (int): The number of unique classes.
        """
        if os.path.splitext(self.classifier_path)[1] == ".pth":
            self.classifier = self.get_classifier_by_path(
                num_columns, unique_count_target
            )
            self.classifier.load_state_dict(torch.load(self.classifier_path))
            self.classifier.eval()
            self.classifier.to(self.device)
        else:
            with open(self.classifier_path, "rb") as file:
                self.classifier = pickle.load(file)

    def get_classifier_by_path(
        self, num_columns: int, unique_count_target: int
    ) -> nn.Module:
        """
        Determine and return the appropriate classifier based on the file path.

        Args:
            num_columns (int): The number of columns in the feature matrix.
            unique_count_target (int): The number of unique classes.

        Returns:
            nn.Module: The classifier model.
        """
        if "_LSTM" in self.classifier_path:
            return LSTM(
                in_features=num_columns, hidden_size=20, num_classes=unique_count_target
            )
        elif "_BasicFFNN" in self.classifier_path:
            return BasicFFNN(num_classes=unique_count_target, in_features=num_columns)
        elif "_IntermediateFFNN" in self.classifier_path:
            return IntermediateFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_AdvancedFFNN" in self.classifier_path:
            return AdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_VeryAdvancedFFNN" in self.classifier_path:
            return VeryAdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        else:
            raise ValueError("Unknown model type in the path")

    def predict(
        self, feature_matrix: pd.DataFrame, mode: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Predict values using the classifier.

        Args:
            feature_matrix (pd.DataFrame): The feature matrix to use for predictions. The index of this DataFrame should contain the enzyme names.
            mode (str): The prediction mode ('metabolism' or 'BGC').

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where each key is an enzyme name, and each value is another dictionary containing the 'predicted_value' and 'score_predicted_values'.
        """
        if mode == "metabolism":
            self.label_encoder.classes_ = np.array(
                ["primary_metabolism", "secondary_metabolism"]
            )
            unique_count_target = 2
        elif mode == "BGC":
            self.label_encoder.classes_ = np.array(BGC_types)
            unique_count_target = len(BGC_types)
        else:
            raise ValueError(f"Mode {mode} not available")

        num_columns = feature_matrix.shape[1]
        logging.debug(f"Num columns: {num_columns}, classes {unique_count_target}")

        # Load classifier if not already loaded
        if self.classifier is None:
            self.load_classifier(num_columns, unique_count_target)

        if isinstance(self.classifier, nn.Module):
            with torch.no_grad():
                feature_matrix_tensor = torch.tensor(
                    feature_matrix.to_numpy(), dtype=torch.float32
                ).to(self.device)
                logits = self.classifier(feature_matrix_tensor)

                predicted_values = torch.argmax(logits, dim=1).cpu().numpy()
                predicted_values = self.label_encoder.inverse_transform(
                    predicted_values
                )
                score_predicted_values = F.softmax(logits, dim=1).cpu().numpy()
        else:
            predicted_values = self.classifier.predict(feature_matrix)
            score_predicted_values = self.classifier.predict_proba(feature_matrix)

        # Creating the result dictionary
        result = {}
        for idx, enzyme_name in enumerate(feature_matrix.index):
            result[enzyme_name] = {
                "predicted_value": predicted_values[idx],
                "score_predicted_values": score_predicted_values[idx],
            }

        return result


class PutativeBGC:
    def __init__(
        self, filtered_dataframe: pd.DataFrame, record: "Record", BGC_type: str
    ):
        """
        Initialize the PutativeBGC class.

        Args:
            filtered_dataframe (pd.DataFrame): DataFrame filtered by window for BGC processing.
            record (Record): The associated Record object containing the GenBank record.
            BGC_type (str): The target BGC type for scoring and categorization.
        """
        self.filtered_dataframe = filtered_dataframe
        self.record = record
        self.start = int(self.filtered_dataframe["cds_start"].min())
        self.end = int(self.filtered_dataframe["cds_end"].max())
        self.BGC_type = BGC_type
        self.score = 0.0

    def calculate_score(self) -> Tuple[float, str]:
        """
        Calculate the score for the BGC.

        Returns:
            Tuple[float, str]: The calculated score and the hybrid BGC type if applicable.
        """
        score = 0
        hybrid_type = None
        types = [["NRPS", "PKS"], ["Terpene", "Alkaloid"]]
        relevant_types = types[0] if self.BGC_type in types[0] else types[1]

        for _, row in self.filtered_dataframe.iterrows():
            bgc_type_score = row["BGC_type_score"]
            np_bgc_affiliation_score = row["NP_BGC_affiliation_score"]

            # If bgc_type_score is a list or array, take the maximum value
            if isinstance(bgc_type_score, (list, np.ndarray)):
                bgc_type_score = max(bgc_type_score)
            # If np_bgc_affiliation_score is a list or array, take the first value
            if isinstance(np_bgc_affiliation_score, (list, np.ndarray)):
                np_bgc_affiliation_score = np_bgc_affiliation_score[0]

            adjusted_score = (bgc_type_score + 0.7) * np_bgc_affiliation_score
            if row["BGC_type"] == self.BGC_type:
                score += adjusted_score
            elif row["BGC_type"] in relevant_types:
                score += adjusted_score / 2
            else:
                score -= adjusted_score

            if (
                row["BGC_type"] in types[0]
                and self.BGC_type in types[0]
                and row["BGC_type"] != self.BGC_type
            ):
                hybrid_type = f"NRPS-PKS-hybrid"

            elif (
                row["BGC_type"] in types[1]
                and self.BGC_type in types[1]
                and row["BGC_type"] != self.BGC_type
            ):
                hybrid_type = f"Terpene-Alkaloid-hybrid"

        # Normalize score based on length and type
        score = (
            (score / max(60_000, self.end - self.start)) * 60_000
            if self.BGC_type in ["NRPS", "PKS"]
            else (score / max(15_000, self.end - self.start)) * 15_000
        )
        self.BGC_type = hybrid_type if hybrid_type else self.BGC_type
        self.score = round(score, 3)
        return self.score, self.BGC_type

    def create_feature(self) -> SeqFeature:
        """
        Create a SeqFeature object for the BGC.

        Returns:
            SeqFeature: A SeqFeature object representing the BGC.
        """
        feature = SeqFeature(
            FeatureLocation(
                start=max(0, self.start), end=min(self.end, len(self.record.record.seq))
            ),
            type="misc_feature",
            qualifiers={
                "label": f"Score: {self.score} {self.BGC_type}",
                "note": "Predicted using Tailenza 1.0.0",
            },
        )
        return feature

    def write_genbank_file(
        self, output_path: str, filename: Optional[str] = None
    ) -> str:
        """
        Write the BGC to a GenBank file.

        Args:
            output_path (str): Directory where the GenBank file will be saved.
            filename (Optional[str]): Optional filename for the GenBank file. If not provided, it will be generated.

        Returns:
            str: The path to the written GenBank file.
        """
        if filename is None:
            filename = (
                f"{self.record.record.id}_{self.start}_{self.end}_{self.score}.gb"
            )

        BGC_record = self.record.record[
            max(0, self.start) : min(self.end, len(self.record.record.seq))
        ]
        BGC_record.annotations["molecule_type"] = "dna"

        output_file = os.path.join(output_path, filename)
        SeqIO.write(BGC_record, output_file, "gb")

        return output_file


class Record:
    def __init__(self, record: SeqRecord, output_dir: str, device: str):
        """
        Initialize the Record class.

        Args:
            record (SeqRecord): The GenBank record.
            output_dir (str): Directory to save output files.
            device (str): Computational device to use (e.g., "cpu", "cuda").
        """
        self.record = record
        self.output_dir = output_dir
        self.device = device
        self.feature_lookup = self.create_feature_lookup()
        self.complete_dataframe = None
        self.tmp_dir = os.path.join(output_dir, "tmp")
        self.tailoring_enzymes = {enzyme: None for enzyme in enzymes}
        self.alignments = {enzyme: None for enzyme in enzymes}
        self.feature_matrixes = {enzyme: None for enzyme in enzymes}
        self.predicted_BGC_types = {enzyme: None for enzyme in enzymes}
        self.scores_predicted_BGC_type = {enzyme: None for enzyme in enzymes}
        self.predicted_metabolism_types = {enzyme: None for enzyme in enzymes}
        self.scores_predicted_metabolism = {enzyme: None for enzyme in enzymes}
        record.complete_dataframe = pd.DataFrame()
        os.makedirs(self.tmp_dir, exist_ok=True)

    def create_feature_lookup(self) -> Dict[str, SeqFeature]:
        """
        Create a lookup dictionary from the record's features.

        Returns:
            Dict[str, SeqFeature]: A dictionary keyed by protein_id or locus_tag.
        """
        feature_lookup = {}
        for feature in self.record.features:
            if feature.type == "CDS":
                protein_id = feature.qualifiers.get(
                    "protein_id", feature.qualifiers.get("locus_tag", ["Unknown"])
                )[0]
                feature_lookup[protein_id] = feature
        return feature_lookup

    def create_putative_bgc(
        self, row: pd.Series, complete_dataframe: pd.DataFrame, BGC_type: str
    ) -> PutativeBGC:
        """
        Create a PutativeBGC object using the calculated window and BGC type.

        Args:
            row (pd.Series): The row of the dataframe representing the feature.
            complete_dataframe (pd.DataFrame): The complete dataframe.
            BGC_type (str): The BGC type to be used for scoring and classification.

        Returns:
            PutativeBGC: An instance of the PutativeBGC class.
        """
        window_start, window_end = self.adjust_window_size(
            row, complete_dataframe, len(self.record.seq)
        )
        filtered_dataframe = complete_dataframe[
            (complete_dataframe["cds_start"] >= window_start)
            & (complete_dataframe["cds_end"] <= window_end)
        ]
        return PutativeBGC(filtered_dataframe, self, BGC_type)

    def adjust_window_size(
        self, row: pd.Series, complete_dataframe: pd.DataFrame, len_record: int
    ) -> Tuple[int, int]:
        """
        Adjust the window size around a feature based on its BGC type.

        Args:
            row (pd.Series): The row of the dataframe representing the feature.
            complete_dataframe (pd.DataFrame): The complete dataframe.
            len_record (int): The length of the sequence record.

        Returns:
            Tuple[int, int]: The start and end of the adjusted window.
        """
        long_BGC_types = ["NRPS", "PKS"]
        short_BGC_types = ["Terpene", "Alkaloid"]

        if row["BGC_type"] in long_BGC_types:
            initial_window = 60000
            trailing_window = 15000
            relevant_types = long_BGC_types
        elif row["BGC_type"] in short_BGC_types:
            initial_window = 15000
            trailing_window = 5000
            relevant_types = short_BGC_types
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
            logging.debug(window_end)
            logging.debug(trailing_window)
            window_start = max(0, window_center - initial_window // 2)
            window_end = window_center + (initial_window // 2)
        else:
            window_end = max(window_start + initial_window, window_end)
        return max(0, window_start - trailing_window), min(
            window_end + trailing_window, len_record
        )

    def calculate_overlap(
        self, start1: int, end1: int, start2: int, end2: int
    ) -> float:
        """
        Calculate the percentage of overlap between two genomic regions.

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
        self,
        score_threshold: float = 0,
    ) -> Dict[str, Dict[str, str]]:
        """
        Process the dataframe, filter based on score, and save results.

        Args:
            complete_dataframe (pd.DataFrame): The complete dataframe with all features.
            score_threshold (float, optional): The minimum score threshold for saving. Defaults to 0.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary of results including BGC information and scores.
        """
        results_dict = {}
        raw_BGCs = []

        for _, row in self.complete_dataframe.iterrows():
            putative_bgc = self.create_putative_bgc(
                row, self.complete_dataframe, row["BGC_type"]
            )
            score, BGC_type = putative_bgc.calculate_score()
            # Normalize score based on length and type
            score = (
                (score / max(60_000, putative_bgc.end - putative_bgc.start)) * 60_000
                if BGC_type in ["NRPS", "PKS"]
                else (score / max(15_000, putative_bgc.end - putative_bgc.start))
                * 15_000
            )
            if score >= score_threshold:
                feature = putative_bgc.create_feature()
                raw_BGCs.append(
                    {
                        "feature": feature,
                        "score": score,
                        "begin": putative_bgc.start,
                        "end": putative_bgc.end,
                        "BGC_type": BGC_type,
                        "putative_bgc": putative_bgc,
                    }
                )
        raw_BGCs.sort(key=lambda x: x["score"], reverse=True)
        filtered_BGCs = []

        for annotation in raw_BGCs:
            overlap = False
            for fa in filtered_BGCs:
                overlap_percent = self.calculate_overlap(
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
        for BGC in filtered_BGCs:
            feature = BGC["feature"]
            self.record.features.append(feature)

            output_path = os.path.join(self.output_dir, BGC["BGC_type"])
            os.makedirs(output_path, exist_ok=True)

            genbank_file_path = BGC["putative_bgc"].write_genbank_file(output_path)
            results_dict[f"{self.record.id}_{BGC['begin']}"] = {
                "ID": self.record.id,
                "description": self.record.description,
                "window_start": BGC["begin"],
                "window_end": BGC["end"],
                "BGC_type": BGC["BGC_type"],
                "score": BGC["score"],
                "filename": os.path.basename(genbank_file_path),
            }

        return results_dict

    def get_tailoring_enzymes(self, hmm_dir: str) -> pd.DataFrame:
        """
        Get tailoring enzymes from the record.

        """
        df = pd.DataFrame()
        for enzyme in enzymes:
            results = self.run_hmmer(enzyme, hmm_dir)

    def run_hmmer(self, enzyme: str, hmm_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Run HMMER to search for enzyme-specific hits in the sequence.

        Args:
            enzyme (str): The enzyme being searched for.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary of features that matched the HMM profile.
        """
        try:
            enzyme_hmm_filename = os.path.join(hmm_dir, enzymes[enzyme]["hmm_file"])
            fasta = os.path.join(self.tmp_dir, f"{self.record.id[:-2]}_temp.fasta")
        except KeyError as e:
            logging.error(f"Key error: {e}")
            return {}

        try:
            self.genbank_to_fasta_cds(fasta)
        except Exception as e:
            logging.error(f"Error in converting GenBank to FASTA: {e}")
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

                            feature = self.feature_lookup.get(hit_name)
                            if feature:
                                results[self.get_identifier(feature)] = (
                                    self.extract_feature_properties(feature)
                                )
                    except Exception as e:
                        logging.error(f"Error during HMMER search: {e}")
        except Exception as e:
            logging.error(f"Error opening or processing files: {e}")

        self.tailoring_enzymes[enzyme] = results
        return results

    def genbank_to_fasta_cds(self, fasta_file: str) -> None:
        """
        Convert GenBank CDS features to a FASTA file.

        Args:
            fasta_file (str): The output FASTA file.
        """
        if os.path.exists(fasta_file):
            return
        with open(fasta_file, "w") as output_handle:
            for feature in self.record.features:
                if feature.type == "CDS":
                    try:
                        protein_id = feature.qualifiers.get(
                            "protein_id",
                            feature.qualifiers.get("locus_tag", ["Unknown"]),
                        )[0]
                        translation = feature.qualifiers.get("translation")[0]
                        sequence = Seq(translation)
                        if 200 <= len(sequence) <= 800:
                            SeqIO.write(
                                SeqRecord(sequence, id=protein_id, description=""),
                                output_handle,
                                "fasta",
                            )
                    except Exception as e:
                        logging.error(
                            f"Error processing feature {feature} in record {self.record.id}: {e}"
                        )
                        continue

    def clear_tmp_dir(self) -> None:
        """
        Clear the temporary directory.
        """
        for filename in os.listdir(self.tmp_dir):
            file_path = os.path.join(self.tmp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")

    def extract_feature_properties(self, feature: SeqFeature) -> Dict[str, str]:
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

    def get_identifier(self, feature: SeqFeature) -> str:
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
        self, product_dict: Dict[str, Dict[str, str]], enzyme_name: str
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

    def set_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def align_sequences(self):
        """
        Align sequences using MUSCLE for all enzymes for all tailoring enzymes.
        """
        for enzyme in enzymes:
            fasta_file = os.path.join(self.tmp_dir, f"{enzyme}_tailoring_enzymes.fasta")

            self.muscle_align_sequences(fasta_filename=fasta_file, enzyme=enzyme)

    def muscle_align_sequences(
        self, fasta_filename: str, enzyme: str
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
            logging.error(f"Failed to run command {' '.join(muscle_cmd)}")
            sys.exit(1)

        alignment = AlignIO.read(open(f"{fasta_filename}_aligned.fasta"), "fasta")
        self.alignments[enzyme] = alignment
        return alignment

    def save_enzymes_to_fasta(self) -> None:
        """Save enzymes together with the reference to a FASTA file."""
        for enzyme, results in self.tailoring_enzymes.items():
            reference_record = SeqRecord(
                Seq(enzymes[enzyme]["reference_for_alignment"]),
                id="Reference",
                description="Reference Sequence",
            )
            seq_records = [reference_record] + [
                SeqRecord(Seq(result["sequence"]), id=id, description=result["product"])
                for id, result in results.items()
            ]
            fasta_name = os.path.join(self.tmp_dir, f"{enzyme}_tailoring_enzymes.fasta")
            SeqIO.write(seq_records, fasta_name, "fasta")

    def safe_map(
        self, row: pd.Series, mapping_dict: Dict[str, str], column_name: str
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

    def featurize_alignments(self, batch_converter, model):
        """
        Featurize the alignments for all enzymes.
        """
        for enzyme, alignment in self.alignments.items():
            if alignment and len(alignment) > 1:
                dataset = AlignmentDataset(enzymes, enzyme, alignment)
                dataset.fragment_alignment(fastas_aligned_before=True)
                feature_matrix = dataset.featurize_fragments(
                    batch_converter, model, self.device
                )
                self.feature_matrixes[enzyme] = feature_matrix
            else:
                self.feature_matrixes[enzyme] = pd.DataFrame()

    def predict_BGC_types(self, directory_of_classifiers_BGC_type: str):
        """
        Predict BGC types for all enzymes.
        """
        for enzyme, feature_matrix in self.feature_matrixes.items():
            if not feature_matrix.empty:
                if enzyme == "ycao":
                    self.predicted_BGC_types[enzyme] = {
                        idx: {"value": "RiPP", "score": [1]}
                        for idx in feature_matrix.index
                    }
                else:
                    classifier_path = os.path.join(
                        directory_of_classifiers_BGC_type,
                        f"{enzyme}{enzymes[enzyme]['classifier_BGC_type']}",
                    )

                    prediction = Predictor(classifier_path, self.device)
                    results = prediction.predict(feature_matrix, mode="BGC")

                    # Store the results directly
                    self.predicted_BGC_types[enzyme] = {
                        idx: {
                            "value": result["predicted_value"],
                            "score": result["score_predicted_values"],
                        }
                        for idx, result in results.items()
                    }
            else:
                self.predicted_BGC_types[enzyme] = {}

    def predict_metabolism_types(self, directory_of_classifiers_NP_affiliation: str):
        """
        Predict metabolism types for all enzymes.
        """
        for enzyme, feature_matrix in self.feature_matrixes.items():
            if not feature_matrix.empty:
                classifier_path = os.path.join(
                    directory_of_classifiers_NP_affiliation,
                    f"{enzyme}{enzymes[enzyme]['classifier_metabolism']}",
                )
                prediction = Predictor(classifier_path, self.device)
                results = prediction.predict(feature_matrix, mode="metabolism")

                # Store the results directly
                self.predicted_metabolism_types[enzyme] = {
                    idx: {
                        "value": result["predicted_value"],
                        "score": result["score_predicted_values"],
                    }
                    for idx, result in results.items()
                }
            else:
                self.predicted_metabolism_types[enzyme] = {}

    def concatenate_results(self):
        """
        Concatenate results from all enzymes into a single DataFrame, including predicted values and scores,
        and merge with tailoring enzymes DataFrame based on the index.
        """
        # Initialize an empty list to hold individual DataFrames
        dataframes = []

        for enzyme, feature_matrix in self.feature_matrixes.items():
            if enzyme not in self.tailoring_enzymes or feature_matrix.empty:
                continue

            # Create a DataFrame for the current enzyme's predictions
            predictions_df = pd.DataFrame(index=feature_matrix.index)
            predictions_df["enzyme"] = enzyme  # Add the enzyme type as a column

            # Populate the DataFrame with BGC type predictions and scores
            predictions_df["BGC_type"] = [
                self.predicted_BGC_types[enzyme].get(idx, {}).get("value", "")
                for idx in feature_matrix.index
            ]
            predictions_df["BGC_type_score"] = [
                self.predicted_BGC_types[enzyme].get(idx, {}).get("score", "")
                for idx in feature_matrix.index
            ]

            # Populate the DataFrame with metabolism type predictions and scores
            predictions_df["NP_BGC_affiliation"] = [
                self.predicted_metabolism_types[enzyme].get(idx, {}).get("value", "")
                for idx in feature_matrix.index
            ]
            predictions_df["NP_BGC_affiliation_score"] = [
                self.predicted_metabolism_types[enzyme].get(idx, {}).get("score", "")
                for idx in feature_matrix.index
            ]

            # Merge the predictions DataFrame with the corresponding tailoring enzymes DataFrame
            tailoring_df = pd.DataFrame.from_dict(
                self.tailoring_enzymes[enzyme], orient="index"
            )
            merged_df = tailoring_df.merge(
                predictions_df, left_index=True, right_index=True, how="inner"
            )

            # Append the merged DataFrame to the list
            dataframes.append(merged_df)

        # Concatenate all the individual DataFrames into one complete DataFrame
        if dataframes:
            self.complete_dataframe = pd.concat(dataframes)
        else:
            self.complete_dataframe = pd.DataFrame()


def main() -> None:
    """
    Main function for processing GenBank files and predicting BGC types.
    """
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
    directory_of_classifiers_NP_affiliation = (
        "../classifiers/classifiers/metabolism_type/"
    )
    fastas_aligned_before = True
    include_charge_features = True
    package_dir = files("tailenza").joinpath("")
    hmm_dir = package_dir.joinpath("data", "HMM_files")
    device = torch.device(args.device[0] if torch.cuda.is_available() else "cpu")

    filename = args.input[0]
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
    try:
        file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(file_path_model)
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()

        for gb_record in SeqIO.parse(filename, "genbank"):
            # Initialize the Record object
            record = Record(gb_record, args.output[0], device)

            # Run HMMER for tailoring enzymes
            record.get_tailoring_enzymes(hmm_dir)
            # Save enzyme sequences to FASTA
            record.save_enzymes_to_fasta()
            record.align_sequences()
            record.featurize_alignments(batch_converter, model)
            # Predict BGC types and metabolism types
            record.predict_BGC_types(directory_of_classifiers_BGC_type)
            record.predict_metabolism_types(directory_of_classifiers_NP_affiliation)
            # Apply predictions to the dataframe
            record.concatenate_results()
            # Process the dataframe and save results
            results_dict = record.process_dataframe_and_save(
                score_threshold=score_threshold,
            )
            # Save results to CSV
            result_df = pd.DataFrame(results_dict)
            result_df.to_csv(
                os.path.join(args.output[0], f"result_dataframe_{gb_record.id}.csv")
            )

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Ensure temporary directory is always cleaned up
        record.clear_tmp_dir()


if __name__ == "__main__":
    main()
