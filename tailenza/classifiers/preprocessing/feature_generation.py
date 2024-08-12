import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import logging
import torch
from typing import List, Tuple, Dict, Optional, Union, Callable

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Helper Functions


def print_gpu_memory() -> None:
    """Logs the GPU memory usage statistics."""
    device = torch.device("cuda")
    logging.info("GPU memory stats:")
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - allocated_memory
    total_memory_gib = total_memory / (1024**3)
    available_memory_gib = available_memory / (1024**3)
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        stats = torch.cuda.memory_stats(device)
        logging.debug(f"{stats}")


def merge_two_dicts(x: Dict, y: Dict) -> Dict:
    """Merges two dictionaries into one.

    Args:
        x (Dict): The first dictionary.
        y (Dict): The second dictionary.

    Returns:
        Dict: A merged dictionary containing keys and values from both input dictionaries.
    """
    z = x.copy()
    z.update(y)
    return z


# Dataset Class


class AlignmentDataset:
    """A class to manage enzyme-related data and alignment processing."""

    def __init__(
        self,
        enzyme_data: Dict[str, Dict],
        enzyme_type: str,
        alignment: List[SeqRecord],
        include_charge_features: bool = True,
    ) -> None:
        """Initializes the Dataset object.

        Args:
            enzyme_data (Dict[str, Dict]): The full enzyme data dictionary.
            enzyme_type (str): The specific enzyme type to use from the dictionary.
            alignment (List[SeqRecord]): A list of alignment records.
        """
        self.enzyme_data = enzyme_data[enzyme_type]
        self.alignment = alignment
        self.splitting_list = self.enzyme_data.get("splitting_list", {})
        self.reference_sequence = self.enzyme_data.get("reference_for_alignment", "")
        self.min_length = self.enzyme_data.get("min_length", 0)
        self.max_length = self.enzyme_data.get("max_length", float("inf"))
        self.modified_splitting_list: Optional[List] = None
        self.fragment_matrix: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.include_charge_features = include_charge_features

    def fragment_alignment(self, fastas_aligned_before: bool = True) -> pd.DataFrame:
        """Splits the alignment into fragments based on a splitting list.

        Args:
            fastas_aligned_before (bool): Whether the sequences are already aligned.

        Returns:
            pd.DataFrame: A DataFrame containing the fragment matrix.
        """
        self.fragment_matrix = pd.DataFrame()
        if not fastas_aligned_before:
            seqa = self.alignment[0]
            seqb = self.alignment[1]
            index_reference = self._indexing_reference(
                SeqRecord(Seq(seqa.seq), id=seqa.id)
            )
            converted_splitting_list = self._convert_splitting_list(
                self.splitting_list, index_reference
            )
            for fragment in converted_splitting_list:
                name_fragment = fragment[0]
                seqRecord_list_per_fragment = self._split_alignment(
                    SeqRecord(Seq(seqb.seq), id=seqb.id),
                    fragment,
                    fastas_aligned_before,
                )
                self.fragment_matrix[name_fragment] = seqRecord_list_per_fragment[:, 1]
                self.fragment_matrix.set_index(
                    pd.Index(seqRecord_list_per_fragment[:, 0])
                )
        else:
            for record in self.alignment:
                if record.id == "Reference":
                    logging.debug("Reference sequence found")
                    index_reference = self._indexing_reference(record)
                    logging.debug("Indexing reference sequence: %s", index_reference)
                    converted_splitting_list = self._convert_splitting_list(
                        self.splitting_list, index_reference
                    )
                    logging.debug(
                        "Converted splitting list: %s", converted_splitting_list
                    )
                    for fragment in converted_splitting_list:
                        name_fragment = fragment[0]
                        seqRecord_list_per_fragment = self._split_alignment(
                            self.alignment, fragment, fastas_aligned_before
                        )
                        self.fragment_matrix[name_fragment] = (
                            seqRecord_list_per_fragment[:, 1]
                        )
                    self.fragment_matrix.index = pd.Index(
                        seqRecord_list_per_fragment[:, 0]
                    )
                    break
        self.fragment_matrix = self._remove_incomplete_rows(self.fragment_matrix)
        self.fragment_matrix["Concatenated"] = (
            self.fragment_matrix.fillna("").astype(str).apply("".join, axis=1)
        )
        return self.fragment_matrix

    def trim_fragment_matrix(
        self, length_threshold: int = 1024, threshold_fragment: int = 100
    ) -> pd.DataFrame:
        """Trims the fragment matrix sequences to fit within a specified length threshold.

        Args:
            length_threshold (int, optional): Maximum allowed length of sequences. Defaults to 1024.
            threshold_fragment (int, optional): Minimum length to consider a fragment excessive. Defaults to 100.

        Returns:
            pd.DataFrame: A trimmed fragment matrix.
        """
        self.fragment_matrix["raw_length"] = (
            self.fragment_matrix.fillna("")
            .astype(str)
            .apply("".join, axis=1)
            .apply(len)
        )
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        logging.debug(f"Length threshold: {length_threshold}")
        logging.debug(f"Threshold fragment: {threshold_fragment}")
        logging.debug(f"Fragment matrix shape: {self.fragment_matrix.shape}")
        long_rows = self.fragment_matrix["raw_length"].gt(length_threshold)
        logging.debug(
            f"Rows exceeding threshold: {self.fragment_matrix['raw_length'].gt(length_threshold).sum(), self.fragment_matrix['raw_length'].max(), self.fragment_matrix[long_rows]}"
        )
        lengths = {
            col: self.fragment_matrix[long_rows][col].astype(str).apply(len).describe()
            for col in self.fragment_matrix.columns[:-1]
        }
        logging.debug(f"Modified column lengths: {lengths}")
        for fragment, (start, end) in self.splitting_list.items():
            normal_length = end - start + 1
            excessive_length = 3 * normal_length
            logging.debug(
                f"Fragment: {fragment}, Normal length: {normal_length}, Excessive length: {excessive_length}"
            )
            self.fragment_matrix[fragment] = self.fragment_matrix.apply(
                lambda row: (
                    row[fragment][:normal_length]
                    if (
                        len(row[fragment]) > excessive_length
                        and len(row[fragment]) > threshold_fragment
                        and row["raw_length"] > length_threshold
                    )
                    else row[fragment]
                ),
                axis=1,
            )
        logging.debug(f"Fragments after handling:  {self.fragment_matrix[long_rows]}")
        lengths = {
            col: self.fragment_matrix[long_rows][col].astype(str).apply(len).describe()
            for col in self.fragment_matrix.columns[:-1]
        }
        logging.debug(f"Modified column lengths: {lengths}")
        self.fragment_matrix["raw_length"] = (
            self.fragment_matrix.fillna("")
            .astype(str)
            .apply(
                lambda row: sum(
                    len(str(row[col])) for col in self.fragment_matrix.columns[:-1]
                ),
                axis=1,
            )
        )
        last_column = self.fragment_matrix.columns[-2]
        self.fragment_matrix[last_column] = self.fragment_matrix.apply(
            lambda row: (
                row[last_column][
                    : len(row[last_column]) - (length_threshold - row["raw_length"])
                ]
                if row["raw_length"] > length_threshold
                else row[last_column]
            ),
            axis=1,
        )
        self.fragment_matrix["raw_length"] = (
            self.fragment_matrix.fillna("")
            .astype(str)
            .apply(
                lambda row: sum(
                    len(str(row[col])) for col in self.fragment_matrix.columns[:-1]
                ),
                axis=1,
            )
        )
        logging.debug(f"Fragments after handling:  {self.fragment_matrix[long_rows]}")
        logging.debug(f"Max raw length: {self.fragment_matrix['raw_length'].max()}")
        assert (
            self.fragment_matrix["raw_length"].max() <= length_threshold
        ), "Error: Some raw_length values exceed the threshold."
        self.fragment_matrix.drop(columns=["raw_length"], inplace=True)
        return self.fragment_matrix

    def featurize_fragments(
        self,
        batch_converter: Callable[
            [List[Tuple[str, str]]], Tuple[List[str], torch.Tensor, torch.Tensor]
        ],
        model: torch.nn.Module,
        device: str = "cpu",
    ) -> Optional[pd.DataFrame]:
        """Generates features for each fragment in the fragment matrix.

        Args:
            batch_converter (function): Function to convert sequences into batches.
            model (torch.nn.Module): The model used to generate embeddings.
            device (str, optional): The device to run the model on. Defaults to "cpu".

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the feature matrix, or None if sequence_strs is empty.
        """
        original_index = self.fragment_matrix.index
        feature_matrix = pd.DataFrame()
        sequence_strs = self.fragment_matrix["Concatenated"].dropna().tolist()
        if not sequence_strs:
            return None
        logging.debug(f"Processing {len(sequence_strs)} sequences ")
        logging.debug(f"Longest sequence: {max(len(s) for s in sequence_strs)}")
        sequence_labels = self.fragment_matrix.index.tolist()
        logging.debug(f"Labels: {sequence_labels}")
        logging.debug(
            f"Type of sequence_strs: {type(sequence_strs)}[{set(type(s) for s in sequence_strs)}]"
        )
        logging.debug(
            f"First item in sequence_strs: {type(sequence_strs[0])}, value: {sequence_strs[0]}"
        )

        logging.debug("Generating embeddings")
        _, _, batch_tokens = batch_converter(list(zip(sequence_labels, sequence_strs)))
        batch_tokens = batch_tokens.to(device)
        num_parts = 4
        batch_size = batch_tokens.size(0)
        part_size = batch_size // num_parts

        token_embeddings_list = []
        for i in range(num_parts):
            start_idx = i * part_size
            end_idx = (i + 1) * part_size if i != num_parts - 1 else batch_size
            part_batch_tokens = batch_tokens[start_idx:end_idx]

            with torch.no_grad():
                results = model(part_batch_tokens, repr_layers=[33])
                logging.debug(f"Results: {results.keys()}")
                token_embeddings_list.append(results["representations"][33])

        token_embeddings = torch.cat(token_embeddings_list, dim=0)
        columns_to_process = self.fragment_matrix.columns.difference(["Concatenated"])

        length_matrix = (
            self.fragment_matrix[columns_to_process]
            .fillna("")
            .apply(lambda col: col.str.len())
        )
        logging.debug(f"Length matrix shape: {length_matrix.shape}")
        logging.debug(f"Length matrix columns: {length_matrix.columns}")
        embedding_means_per_fragment = self._fragment_means(
            token_embeddings, length_matrix
        )
        logging.debug(f"Embedding means shape: {len(embedding_means_per_fragment)}")
        embedding_df = self._convert_embeddings_to_dataframe(
            embedding_means_per_fragment, original_index, length_matrix.columns
        )
        if self.include_charge_features:
            # Refactored from applymap(self._calculate_charge) to apply with a lambda function
            charge_matrix = self.fragment_matrix.fillna("").apply(
                lambda col: col.apply(self._calculate_charge)
            )
            charge_matrix.columns = ["charge_" + col for col in charge_matrix.columns]
        if self.include_charge_features:
            feature_matrix = pd.concat([embedding_df, charge_matrix], axis=1)
        else:
            feature_matrix = embedding_df
        logging.debug(f"Embedding means shape: {len(embedding_means_per_fragment)}")

        logging.debug(f"Feature matrix shape: {feature_matrix.shape}")
        logging.debug(f"Feature matrix columns: {feature_matrix.columns}")
        logging.debug(f"Feature matrix head: {feature_matrix.head()}")
        if self.fragment_matrix.isnull().values.any():
            raise ValueError("Fragment matrix contains NaN values.")

        self.feature_matrix = feature_matrix
        return self.feature_matrix

    def _split_alignment(
        self,
        alignment: Union[List[SeqRecord], SeqRecord],
        fragment: List[Union[str, int]],
        fastas_aligned_before: bool,
    ) -> np.ndarray:
        """Splits an alignment into sub-sequences based on a fragment.

        Args:
            alignment (Union[List[SeqRecord], SeqRecord]): List of aligned sequences or a single sequence.
            fragment (List[Union[str, int]]): A list containing the fragment name and its start and end positions.
            fastas_aligned_before (bool): Whether the sequences are already aligned.

        Returns:
            np.ndarray: An array of sequences corresponding to the fragment.
        """
        start = fragment[1]
        end = fragment[2]
        if not fastas_aligned_before:
            alignment = [alignment]
        logging.debug("Splitting alignment")
        seqRecord_list_per_fragment = []
        if fragment[0] == "begin":
            start = 1
        if fragment[0] != "end":
            for record in alignment:
                if record.id != "Reference":
                    subsequence = str(record.seq)[start - 1 : end - 1].replace("-", "")
                    seqRecord_list_per_fragment.append([record.id, subsequence])
        else:
            for record in alignment:
                if record.id != "Reference":
                    subsequence = str(record.seq)[start - 1 :].replace("-", "")
                    seqRecord_list_per_fragment.append([record.id, subsequence])
        seqRecord_array_per_fragment = np.array(seqRecord_list_per_fragment)
        return seqRecord_array_per_fragment

    def _remove_incomplete_rows(self, fragment_matrix: pd.DataFrame) -> pd.DataFrame:
        """Removes rows from the fragment matrix that contain empty fragments.

        Args:
            fragment_matrix (pd.DataFrame): The fragment matrix.

        Returns:
            pd.DataFrame: The fragment matrix with incomplete rows removed.
        """
        initial_row_count = len(fragment_matrix)
        fragment_matrix.replace("", pd.NA, inplace=True)
        fragment_matrix = fragment_matrix.dropna(how="any")
        final_row_count = len(fragment_matrix)
        dropped_rows = initial_row_count - final_row_count
        logging.info(f"Rows dropped due to empty fragments: {dropped_rows}")
        return fragment_matrix

    def _indexing_reference(self, record: SeqRecord) -> List[Tuple[int, int]]:
        """Creates an index mapping for the reference sequence without ignoring gaps.

        Args:
            record (SeqRecord): The reference sequence record.

        Returns:
            List[Tuple[int, int]]: A list of tuples containing the amino acid index and its position.
        """
        list_reference = list(str(record.seq))
        index_aa = 0
        index_mapping = []
        for index, AA in enumerate(list_reference):
            if AA != "-":
                index_aa += 1
                index_mapping.append((index_aa, index))
        return index_mapping

    def _convert_splitting_list(
        self,
        splitting_list: Dict[str, Tuple[int, int]],
        index_reference: List[Tuple[int, int]],
    ) -> List[Tuple[str, int, int]]:
        """Converts the canonical splitting list to reflect gaps in the reference sequence.

        Args:
            splitting_list (Dict[str, Tuple[int, int]]): The canonical splitting list.
            index_reference (List[Tuple[int, int]]): The index mapping for the reference sequence.

        Returns:
            List[Tuple[str, int, int]]: A converted splitting list that accounts for gaps in the reference sequence.
        """
        converted_splitting_list = []
        for fragment, (begin, end) in splitting_list.items():
            converted_splitting_list.append(
                (
                    fragment,
                    index_reference[begin - 1][1],
                    index_reference[end - 1][1],
                )
            )
        return converted_splitting_list

    def _calculate_charge(self, sequence: str) -> float:
        """Calculates the approximate charge of an amino acid sequence.

        Args:
            sequence (str): The amino acid sequence.

        Returns:
            float: The calculated charge of the sequence.
        """
        AACharge = {
            "C": -0.045,
            "D": -0.999,
            "E": -0.998,
            "H": 0.091,
            "K": 1,
            "R": 1,
            "Y": -0.001,
        }
        charge = -0.002
        seqstr = str(sequence)
        seqlist = list(seqstr)
        for aa in seqlist:
            if aa in AACharge:
                charge += AACharge[aa]
        return charge / 10

    def _convert_embeddings_to_dataframe(
        self, embeddings: List[torch.Tensor], index: pd.Index, fragments: List[str]
    ) -> Optional[pd.DataFrame]:
        """Converts a list of embedding tensors into a pandas DataFrame.

        Args:
            embeddings (List[torch.Tensor]): List of embedding tensors.
            index (pd.Index): Index for the resulting DataFrame.
            fragments (List[str]): List of fragment names.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with the embeddings flattened and organized by fragment, or None if the conversion fails.
        """
        data = [tensor.cpu().numpy() for tensor in embeddings]
        total_features = data[0].shape[0]
        if total_features % len(fragments) != 0:
            logging.error(
                "Total number of features is not evenly divisible by the number of fragments."
            )
            return None
        num_features_per_fragment = total_features // len(fragments)
        columns = [
            f"{fragment}_{i}"
            for fragment in fragments
            for i in range(num_features_per_fragment)
        ]
        try:
            df = pd.DataFrame(data, index=index, columns=columns)
        except Exception as e:
            logging.error(f"Failed to create DataFrame: {e}")
            logging.error(f"Data shape: {len(data)}, {len(data[0])}")
            logging.error(f"Data: {data}")
            return None
        return df

    def _filter_alignment(self) -> List[SeqRecord]:
        """Filters the alignment based on the length of the sequences without gaps.

        This method iterates through the sequences in the alignment and selects
        those whose length (excluding gaps) falls within the specified minimum
        and maximum length.

        Returns:
            List[SeqRecord]: A list of SeqRecord objects that meet the length criteria.
        """
        filtered_alignment: List[SeqRecord] = []
        for record in self.alignment:
            pure_length = len(str(record.seq).replace("-", ""))
            if self.min_length <= pure_length <= self.max_length:
                filtered_alignment.append(record)
        self.alignment = filtered_alignment
        return filtered_alignment

    def _fragment_means(
        self, embeddings: torch.Tensor, lengths: pd.DataFrame
    ) -> List[torch.Tensor]:
        """Calculates the mean embeddings for each fragment.

        Args:
            embeddings (torch.Tensor): The embeddings tensor.
            lengths (pd.DataFrame): DataFrame containing the lengths of each fragment.

        Returns:
            List[torch.Tensor]: A list of mean embeddings for each fragment.
        """
        fragment_results = []
        for embedding, length_row in zip(embeddings, lengths.itertuples(index=False)):
            start = 0
            means = []
            logging.debug(f"Embedding length: {len(embedding[0])}")
            for length in length_row:
                if length == 0:
                    means.append(
                        torch.zeros(
                            embedding.size(1),
                            dtype=embedding.dtype,
                            device=embedding.device,
                        )
                    )
                else:
                    fragment_embedding = embedding[start : start + length]
                    if fragment_embedding.nelement() == 0:
                        means.append(
                            torch.zeros(
                                embedding.size(1),
                                dtype=embedding.dtype,
                                device=embedding.device,
                            )
                        )
                    else:
                        fragment_mean = fragment_embedding.mean(dim=0)
                        means.extend(fragment_mean)
                start += length
            fragment_results.append(torch.stack(means))
        logging.debug(f"Fragment results shape: {fragment_results[0].shape}")
        logging.debug(f"Fragment results length: {len(fragment_results)}")
        return fragment_results
