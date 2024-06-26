#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 08:26:52 2022

@author: friederike
"""


import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

def print_gpu_memory():
    device = torch.device("cuda")
    logging.info("GPU memory stats:")
    # Get GPU properties
    props = torch.cuda.get_device_properties(device)
    
    # Total memory in bytes
    total_memory = props.total_memory

    # Allocated memory in bytes
    allocated_memory = torch.cuda.memory_allocated(device)

    # Available memory in bytes
    available_memory = total_memory - allocated_memory

    # Convert to GiB for readability
    total_memory_gib = total_memory / (1024 ** 3)
    available_memory_gib = available_memory / (1024 ** 3)

    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        stats = torch.cuda.memory_stats(device)
        logging.debug(f"{stats}")

def filter_alignment(alignment, min_length, max_length):
    # filter the alignment based on the length of the sequences
    filtered_alignment = []
    for record in alignment:
        pure_length = len(str(record.seq).replace("-", ""))
        if pure_length >= min_length and pure_length <= max_length:
            filtered_alignment.append(record)
    return filtered_alignment


def merge_two_dicts(x, y):
    # input-> 2 dictionaries output->  merged dictionary
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def calculate_charge(sequence):
    # uses aa sequence as input and calculates the approximate charge of it
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
    return charge / 10  # Add some normalization


def easysequence(sequence):
    # creates a string out of the sequence file, that only states if AA is acidic (a), basic (b), polar (p), neutral/unpolar (n),aromatic (r),Cystein (s) or a Prolin (t)
    seqstr = str(sequence)
    seqlist = list(seqstr)
    easylist = []
    for i in seqlist:
        if i == "E" or i == "D":
            easylist = easylist + ["a"]
        if i == "K" or i == "R" or i == "H":
            easylist = easylist + ["b"]
        if i == "S" or i == "T" or i == "N" or i == "Q":
            easylist = easylist + ["p"]
        if i == "F" or i == "Y" or i == "W":
            easylist = easylist + ["r"]
        if i == "C":
            easylist = easylist + ["s"]
        if i == "P":
            easylist = easylist + ["t"]
        if i == "G" or i == "A" or i == "V" or i == "L" or i == "I" or i == "M":
            easylist = easylist + ["n"]

    seperator = ""
    easysequence = seperator.join(easylist)
    return easysequence


def indexing_reference(record):
    # index the reference sequence without ignoring gaps
    list_reference = list(str(record.seq))
    index_aa = 0
    index_mapping = []
    for index, AA in enumerate(list_reference):
        if AA != "-":
            index_aa += 1
            index_mapping.append([index_aa, index])

    return index_mapping


def convert_splitting_list(splitting_list: dict, index_reference):
    # -> convert the canonic splitting list to also reflect eventual gaps in the reference sequence
    converted_splitting_list = []

    for fragment, [begin, end] in splitting_list.items():
        converted_splitting_list.append(
            [
                fragment,
                index_reference[begin - 1][1],
                index_reference[end - 1][1],
            ]
        )
    return converted_splitting_list


def split_alignment(alignment, fragment, fastas_aligned_before):
    # split the aligned sequences at the positions determined by the splitting list
    start = fragment[1]
    end = fragment[2]
    if fastas_aligned_before == False:
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


def remove_incomplete_rows(fragment_matrix: pd.DataFrame) -> pd.DataFrame:
    # Capture initial row count
    initial_row_count = len(fragment_matrix)

    # Drop any rows where any fragment is empty
    fragment_matrix.replace("", pd.NA, inplace=True)
    fragment_matrix = fragment_matrix.dropna(how="any")

    # Capture new row count and calculate the number of dropped rows
    final_row_count = len(fragment_matrix)
    dropped_rows = initial_row_count - final_row_count
    logging.info(f"Rows dropped due to empty fragments: {dropped_rows}")

    return fragment_matrix


def trim_fragment_matrix(
    fragment_matrix,
    splitting_list,
    length_threshold: int = 1024,
    threshold_fragment: int = 100,
) -> pd.DataFrame:
    # trim sequences longer than the threshold to allow transformer processing
    fragment_matrix["raw_length"] = (
        fragment_matrix.fillna("").astype(str).apply("".join, axis=1).apply(len)
    )
    pd.set_option("display.max_rows", None)  # Set to None to display all rows
    pd.set_option("display.max_columns", None)  # Set to None to display all columns
    logging.debug(f"Length threshold: {length_threshold}")
    logging.debug(f"Threshold fragment: {threshold_fragment}")
    logging.debug(f"Fragment matrix shape: {fragment_matrix.shape}")
    long_rows = fragment_matrix["raw_length"].gt(length_threshold)
    logging.debug(
        f"Rows exceeding threshold: {fragment_matrix['raw_length'].gt(length_threshold).sum(), fragment_matrix['raw_length'].max(), fragment_matrix[long_rows]}"
    )
    lengths = {
        col: fragment_matrix[long_rows][col]
        .astype(str)
        .apply(len)
        .describe()  # Convert to string before applying len()
        for col in fragment_matrix.columns[:-1]  # Excluding 'raw_length'
    }
    logging.debug(f"Modified column lengths: {lengths}")
    for fragment, (start, end) in splitting_list.items():
        normal_length = end - start + 1  # Calculate the normal length of the fragment
        excessive_length = 3 * normal_length  # Calculate 3 times the normal length
        logging.debug(
            f"Fragment: {fragment}, Normal length: {normal_length}, Excessive length: {excessive_length}"
        )

        # If a fragment is 3* longer than usual and longer than 100 and longer than threshold, trim it
        fragment_matrix[fragment] = fragment_matrix.apply(
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
    logging.debug(f"Fragments after handling:  {fragment_matrix[long_rows]}")
    lengths = {
        col: fragment_matrix[long_rows][col]
        .astype(str)
        .apply(len)
        .describe()  # Convert to string before applying len()
        for col in fragment_matrix.columns[:-1]  # Excluding 'raw_length'
    }
    logging.debug(f"Modified column lengths: {lengths}")
    # Recalculate raw_length after all modifications
    fragment_matrix["raw_length"] = (
        fragment_matrix.fillna("")
        .astype(str)
        .apply(
            lambda row: sum(len(str(row[col])) for col in fragment_matrix.columns[:-1]),
            axis=1,
        )
    )
    # Check that after processing, the lengths are ok, if not, trim the sequences to the threshold
    last_column = fragment_matrix.columns[-2]
    fragment_matrix[last_column] = fragment_matrix.apply(
        lambda row: (
            row[last_column][
                : len(row[last_column]) - (length_threshold - row["raw_length"])
            ]
            if row["raw_length"] > length_threshold
            else row[last_column]
        ),
        axis=1,
    )
    # Recalculate raw_length after all modifications
    fragment_matrix["raw_length"] = (
        fragment_matrix.fillna("")
        .astype(str)
        .apply(
            lambda row: sum(len(str(row[col])) for col in fragment_matrix.columns[:-1]),
            axis=1,
        )
    )
    logging.debug(f"Fragments after handling:  {fragment_matrix[long_rows]}")
    logging.debug(f"Max raw length: {fragment_matrix['raw_length'].max()}")
    assert (
        fragment_matrix["raw_length"].max() <= length_threshold
    ), "Error: Some raw_length values exceed the threshold."
    fragment_matrix.drop(columns=["raw_length"], inplace=True)
    return fragment_matrix


def fragment_alignment(alignment, splitting_list, fastas_aligned_before):
    # create a matrix from the splitted alignment
    fragment_matrix = pd.DataFrame()
    if fastas_aligned_before == False:

        seqa = alignment[0]
        seqb = alignment[1]
        index_reference = indexing_reference(SeqRecord(Seq(seqa), id=seqa))

        converted_splitting_list = convert_splitting_list(
            splitting_list, index_reference
        )
        for fragment in converted_splitting_list:
            name_fragment = fragment[0]
            seqRecord_list_per_fragment = split_alignment(
                SeqRecord(Seq(seqb), id=seqb), fragment, fastas_aligned_before
            )

            fragment_matrix[name_fragment] = seqRecord_list_per_fragment[:, 1]
            fragment_matrix.set_index(pd.Index(seqRecord_list_per_fragment[:, 0]))
    else:
        for record in alignment:
            if record.id == "Reference":
                logging.debug("Reference sequence found")
                index_reference = indexing_reference(record)
                logging.debug("Indexing reference sequence", index_reference)
                converted_splitting_list = convert_splitting_list(
                    splitting_list, index_reference
                )
                logging.debug("Converted splitting list", converted_splitting_list)
                for fragment in converted_splitting_list:
                    name_fragment = fragment[0]
                    seqRecord_list_per_fragment = split_alignment(
                        alignment, fragment, fastas_aligned_before
                    )
                    fragment_matrix[name_fragment] = seqRecord_list_per_fragment[:, 1]
                fragment_matrix.index = pd.Index(seqRecord_list_per_fragment[:, 0])
                break
    fragment_matrix = remove_incomplete_rows(fragment_matrix)
    # fragment_matrix = trim_fragment_matrix(fragment_matrix, splitting_list)

    fragment_matrix["Concatenated"] = (
        fragment_matrix.fillna("").astype(str).apply("".join, axis=1)
    )
    return fragment_matrix


def fragment_means(embeddings, lengths) -> list:
    fragment_results = []
    print_gpu_memory()
    for embedding, length_row in zip(embeddings, lengths.itertuples(index=False)):
        start = 0
        means = []
        # Log the length of embedding
        logging.debug(f"Embedding length: {len(embedding[0])}")
        # Calculate mean for each fragment based on length
        for length in length_row:
            if length == 0:
                # Ensure the zero tensor matches the expected dimensions of fragment embeddings
                means.extend(
                    torch.zeros(
                        embedding.size(1),
                        dtype=embedding.dtype,
                        device=embedding.device,
                    )
                )
            else:
                fragment_embedding = embedding[start : start + length]
                # Safeguard against empty slices which should not occur if lengths are correct
                if fragment_embedding.nelement() == 0:
                    means.extend(
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
        # Collect means for all fragments in the row
        fragment_results.append(torch.stack(means))
    logging.debug(f"Fragment results shape: {fragment_results[0].shape}")
    logging.debug(f"Fragment results length: {len(fragment_results)}")
    print_gpu_memory()
    return fragment_results


def convert_embeddings_to_dataframe(embeddings, index, fragments):
    """
    Convert list of embedding tensors into a DataFrame.
    """
    # Flatten each tensor and convert to numpy
    logging.debug(f"Embeddings shape: {len(embeddings)}")
    logging.debug(f"Embedding shape: {embeddings[0].shape}")
    data = [tensor.cpu().numpy() for tensor in embeddings]
    logging.debug(f"Data shape: {data[0].shape}")
    # Check total number of features
    total_features = data[0].shape[0]
    # Calculate number of features per fragment
    if total_features % len(fragments) != 0:
        logging.error(
            "Total number of features is not evenly divisible by the number of fragments."
        )
        return None

    num_features_per_fragment = total_features // len(fragments)

    # Create column names
    columns = [
        f"{fragment}_{i}"
        for fragment in fragments
        for i in range(num_features_per_fragment)
    ]

    # Create DataFrame
    try:
        df = pd.DataFrame(data, index=index, columns=columns)
    except Exception as e:
        logging.error(f"Failed to create DataFrame: {e}")
        return None

    return df


def featurize_fragments(
    fragment_matrix: pd.DataFrame,
    batch_converter,
    model,
    include_charge_features: bool = True,
    device: str = "cpu",
):
    """
    Generate features for each fragment in the fragments dictionary.
    """
    original_index = fragment_matrix.index
    feature_matrix = pd.DataFrame()
    sequence_strs = (
        fragment_matrix["Concatenated"].dropna().tolist()
    )  # Ensure to drop any NaN values
    if not sequence_strs:
        return None
    logging.debug(f"Processing {len(sequence_strs)} sequences ")
    logging.debug(f"Longest sequence: {max(len(s) for s in sequence_strs)}")
    sequence_labels = fragment_matrix.index.tolist()
    logging.debug(f"Labels: {sequence_labels}")
    logging.debug(
        f"Type of sequence_strs: {type(sequence_strs)}[{set(type(s) for s in sequence_strs)}]"
    )
    logging.debug(
        f"First item in sequence_strs: {type(sequence_strs[0])}, value: {sequence_strs[0]}"
    )

    # Generate embeddings
    logging.debug("Generating embeddings")
    _, _, batch_tokens = batch_converter(list(zip(sequence_labels, sequence_strs)))
    batch_tokens = batch_tokens.to(device)
    # Number of parts to split the batch into
    #for esm-2
    #num_parts = len(sequence_strs)
    num_parts = 4
    batch_size = batch_tokens.size(0)
    part_size = batch_size // num_parts
    
    # Initialize an empty list to store the token embeddings for each part
    token_embeddings_list = []
    
    # Process each part of the batch
    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = (i + 1) * part_size if i != num_parts - 1 else batch_size
        part_batch_tokens = batch_tokens[start_idx:end_idx]

        with torch.no_grad():
            results = model(part_batch_tokens, repr_layers=[33])
            logging.debug(f"Results: {results.keys()}")
            token_embeddings_list.append(results["representations"][33])

    # Concatenate the embeddings
    token_embeddings = torch.cat(token_embeddings_list, dim=0)
    # Select columns excluding 'Concatenated'
    columns_to_process = fragment_matrix.columns.difference(["Concatenated"])

    # Calculate the length of each entry in the selected columns
    length_matrix = fragment_matrix[columns_to_process].fillna("").applymap(len)
    logging.debug(f"Length matrix shape: {length_matrix.shape}")
    logging.debug(f"Length matrix columns: {length_matrix.columns}")
    embedding_means_per_fragment = fragment_means(token_embeddings, length_matrix)
    logging.debug(f"Embedding means shape: {len(embedding_means_per_fragment)}")
    embedding_df = convert_embeddings_to_dataframe(
        embedding_means_per_fragment, original_index, length_matrix.columns
    )
    if include_charge_features:
        charge_matrix = fragment_matrix.fillna("").applymap(calculate_charge)
        charge_matrix.columns = ["charge_" + col for col in charge_matrix.columns]
    if include_charge_features:
        feature_matrix = pd.concat([embedding_df, charge_matrix], axis=1)
    else:
        feature_matrix = embedding_df
    logging.debug(f"Embedding means shape: {len(embedding_means_per_fragment)}")

    logging.debug(f"Feature matrix shape: {feature_matrix.shape}")
    logging.debug(f"Feature matrix columns: {feature_matrix.columns}")
    logging.debug(f"Feature matrix head: {feature_matrix.head()}")
    if fragment_matrix.isnull().values.any():
        raise ValueError("Fragment matrix contains NaN values.")

    return feature_matrix
