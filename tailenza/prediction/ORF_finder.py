#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

# Configuration
output_fasta = "orfs.fasta"
output_protein_fasta = "orfs_protein.fasta"

# Define a list of potential RBS patterns
rbs_patterns = ["AGGAGG", "GGAGG", "AGGAG", "GAGG", "AGGA"]
alternative_start_codons = ["ATG", "GTG", "TTG"]
stop_codons = ["TAA", "TAG", "TGA"]

def search_orfs(sequence, is_reverse=False):
    found_orfs = []
    seq_length = len(sequence)
    for rbs_pattern in rbs_patterns:
        for rbs_start in range(seq_length - len(rbs_pattern)):
            if sequence[rbs_start:rbs_start + len(rbs_pattern)] == rbs_pattern:
                for start_codon in range(rbs_start + len(rbs_pattern), min(rbs_start + len(rbs_pattern) + 20, seq_length - 3)):
                    if sequence[start_codon:start_codon + 3] in alternative_start_codons:
                        for end_codon in range(start_codon + 30, seq_length, 3):
                            codon = sequence[end_codon:end_codon + 3]
                            if codon in stop_codons:
                                orf_seq = Seq(sequence[start_codon:end_codon + 3])
                                protein_seq = orf_seq.translate(to_stop=True)
                                if 10 <= len(protein_seq) <= 100:
                                    if '*' not in str(protein_seq):
                                        strand = "-" if is_reverse else "+"
                                        start_pos = seq_length - end_codon - 3 if is_reverse else start_codon
                                        end_pos = seq_length - start_codon if is_reverse else end_codon + 3
                                        found_orfs.append((orf_seq, protein_seq, strand, start_pos, end_pos))
                                    break
    return found_orfs

def find_orfs_with_criteria(dna_sequence):
    found_orfs = []
    orfs_forward = search_orfs(dna_sequence)
    reverse_complement = str(Seq(dna_sequence).reverse_complement())
    orfs_reverse = search_orfs(reverse_complement, is_reverse=True)
    found_orfs = orfs_forward + orfs_reverse
    return found_orfs

def process_gbk_file(gbk_file):
    orfs_found = []
    updated_records = []
    for record in SeqIO.parse(gbk_file, "genbank"):
        dna_sequence = str(record.seq)
        found_orfs = find_orfs_with_criteria(dna_sequence)
        for orf_seq, protein_seq, strand, start_pos, end_pos in found_orfs:
            overlapping = False
            for feature in record.features:
                if feature.type == "CDS" and len(feature.location) > 150:
                    if not (end_pos <= feature.location.start or start_pos >= feature.location.end):
                        overlapping = True
                        break
            if not overlapping:
                orf_feature = SeqFeature(
                    FeatureLocation(start_pos, end_pos, strand=1 if strand == "+" else -1),
                    type="CDS",
                    qualifiers={"translation": str(protein_seq)}
                )
                record.features.append(orf_feature)
                orfs_found.append((orf_seq, protein_seq, strand, start_pos, end_pos, record.id))
        updated_records.append(record)
    return orfs_found, updated_records

def main():
    if len(sys.argv) != 2:
        print("Usage: script.py <input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]

    if not os.path.isdir(input_directory):
        print(f"Error: {input_directory} is not a valid directory")
        sys.exit(1)

    # Process all GBK files and write ORFs to a FASTA file
    seen_orfs = set()
    with open(output_fasta, "w") as fasta_output, open(output_protein_fasta, "w") as protein_fasta_output:
        for gbk_file in os.listdir(input_directory):
            if gbk_file.endswith(".gbk"):
                gbk_path = os.path.join(input_directory, gbk_file)
                orfs, updated_records = process_gbk_file(gbk_path)
                for i, (orf_seq, protein_seq, strand, start_pos, end_pos, record_id) in enumerate(orfs, 1):
                    orf_key = (str(orf_seq), start_pos, end_pos, record_id)
                    if orf_key not in seen_orfs:
                        fasta_output.write(f">{record_id}_ORF_{i}_strand_{strand}_pos_{start_pos}-{end_pos}\n{orf_seq}\n")
                        protein_fasta_output.write(f">{record_id}_ORF_{i}_strand_{strand}_pos_{start_pos}-{end_pos}\n{protein_seq}\n")
                        seen_orfs.add(orf_key)
                new_gbk_path = os.path.join(input_directory, gbk_file.replace(".gbk", "_with_orfs.gbk"))
                with open(new_gbk_path, "w") as output_handle:
                    SeqIO.write(updated_records, output_handle, "genbank")

if __name__ == "__main__":
    main()
