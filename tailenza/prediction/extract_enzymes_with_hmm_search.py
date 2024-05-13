import os
import sys
import subprocess
import shutil
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from feature_generation import *
from enzyme_information import enzymes
import pickle
import matplotlib.pyplot as plt
import argparse
import pyhmmer
import warnings
from Bio import BiopythonWarning
from subprocess import DEVNULL
from enzyme_information import enzymes


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


def run_hmmer(record, enzyme):
    try:
        enzyme_hmm_filename = os.path.join("HMM_files", enzymes[enzyme]["hmm_file"])
        fasta = os.path.join(tmp_dir, f"{record.id}_temp.fasta")
    except KeyError as e:
        print(f"Key error: {e}")
        return {}

    try:
        genbank_to_fasta_cds(record, fasta)
    except Exception as e:
        print(f"Error in converting GenBank to FASTA: {e}")
        return {}

    try:
        feature_lookup = create_feature_lookup(record)
    except Exception as e:
        print(f"Error in creating feature lookup: {e}")
        return {}

    results = {}

    if not os.path.exists(enzyme_hmm_filename):
        print(f"HMM file {enzyme_hmm_filename} does not exist.")
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
                            results[
                                get_identifier(feature)
                            ] = extract_feature_properties(feature)
                except Exception as e:
                    print(f"Error during HMMER search: {e}")
    except Exception as e:
        print(f"Error opening or processing files: {e}")
    return results


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
                    description = feature.qualifiers.get("product", ["Unknown"])[0]
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
                    # Create a new SeqRecord and write to the output handle
                    SeqIO.write(
                        SeqIO.SeqRecord(
                            sequence, id=protein_id, description=description
                        ),
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


def save_enzymes_to_fasta(all_enzymes_results):
    for enzyme, results in all_enzymes_results.items():
        seq_records = []
        for record_id, record_results in results.items():
            gbk_filename = record_id.split("_")[
                0
            ]  # Extract the filename from the record ID
            for enzyme_id, properties in record_results.items():
                description = f"{gbk_filename} | {enzyme_id} | {properties['product']} | {properties['cds_start']}-{properties['cds_end']}"
                seq_records.append(
                    SeqRecord(
                        Seq(properties["sequence"]),
                        id=enzyme_id,
                        description=description,
                    )
                )

        fasta_name = os.path.join(tmp_dir, f"{enzyme}_tailoring_enzymes.fasta")
        SeqIO.write(seq_records, fasta_name, "fasta")


# Directory containing GenBank files
genbank_dir = "/home/friederike/Documents/Coding/TailEnzA_main/TailEnzA/Prediction/Prediciton_21_11_2023/Tailenza_score_over_1/GenBank_files_over_1_updated_plot_precursors_added_with_ripp_like/"
tmp_dir = "/home/friederike/Documents/Coding/TailEnzA_main/TailEnzA/Prediction/Prediciton_21_11_2023/Tailenza_score_over_1/genbank_without_biotin_with_ripp_like_ssn_threshold_15/fastas_for_ssn/"

all_enzymes_results = {enzyme: {} for enzyme in enzymes}

# Iterate over each GenBank file
for genbank_file in os.listdir(genbank_dir):
    if genbank_file.endswith(".gbk") or genbank_file.endswith(".gb"):
        record_path = os.path.join(genbank_dir, genbank_file)
        for record in SeqIO.parse(record_path, "genbank"):
            record_id = f"{genbank_file}_{record.id}"
            # Perform HMM search for each enzyme
            for enzyme in enzymes:
                results = run_hmmer(record, enzyme)
                if results:
                    if enzyme not in all_enzymes_results:
                        all_enzymes_results[enzyme] = {}
                    all_enzymes_results[enzyme][record_id] = results

# Save results to FASTA files for all enzymes
save_enzymes_to_fasta(all_enzymes_results)
