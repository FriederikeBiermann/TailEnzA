from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import argparse
import pyhmmer

# Create Argument Parser for command line arguments
parser = argparse.ArgumentParser(description="HMMER search on input FASTA file.")
parser.add_argument('input_fasta', help='Path to the input FASTA file')
parser.add_argument('pfam_hmm_database', help='Path to PFAM HMM database')
parser.add_argument('output_path', help='Path to save HMMER search results')
args = parser.parse_args()


enzymes = ["ycao", "P450", "radical SAM", "Methyltransf_2", "TP_methylase", "Methyltranf_3", "Methyltransf_25"]

# Load sequences from the input fasta file
all_sequences = list(SeqIO.parse(args.input_fasta, "fasta"))
sequences_dict = {seq.id: seq for seq in all_sequences}

# HMMER Search on Loaded Sequences

for enzyme in enzymes:
    enzyme_hmm_filename = f"{enzyme}.hmm"
    file_path = os.path.join(args.pfam_hmm_database, enzyme_hmm_filename)

    if os.path.exists(file_path):
        output_file = os.path.join(args.output_path, f"NCBI_{enzyme}_hmmer.fasta")

        with pyhmmer.easel.SequenceFile(args.input_fasta, digital=True) as seq_file:
            sequences = seq_file.read_block()
            with open(output_file, "w") as output_fasta_file:
                with pyhmmer.plan7.HMMFile(file_path) as hmm_file:
                    for hits in pyhmmer.hmmsearch(hmm_file, sequences, cpus=4):
                        for hit in hits:
                            evalue = hit.evalue
                            hit_name = hit.name.decode()

                            if evalue >= 10e-20:
                                continue
                            record = sequences_dict.get(hit_name)
                            if record:
                                SeqIO.write(record, output_fasta_file, "fasta")
                print(f"Hits saved to {output_file}")

    else:
        print(f"Enzyme HMM file '{enzyme_hmm_filename}' not found in provided database.")
