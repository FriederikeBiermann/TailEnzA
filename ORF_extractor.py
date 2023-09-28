import sys
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation


def translate_orfs(gb_file):
    # Load the GenBank file
    record = SeqIO.read(gb_file, "genbank")

    # Extract the nucleotide sequences of all features called "ORF"
    orfs = [feat for feat in record.features if feat.type == "ORF"]
    sequences = []
    for orf in orfs:
        # Parse the feature location
        location = orf.location

        # Extract the nucleotide sequence
        if location.strand == -1:
            orf_seq = Seq(str(record.seq[location.start:location.end].reverse_complement().translate(table=11)))
        else:
            orf_seq = Seq(str(record.seq[location.start:location.end].translate(table=11)))
        sequences.append(orf_seq)

    # Create SeqRecord objects for each ORF sequence
    orf_records = [SeqRecord(seq, id="{}_ORF{}".format(record.id, i+1), description="") for i, seq in enumerate(sequences)]

    # Save the sequences as a FASTA file
    output_file = os.path.splitext(gb_file)[0] + "_orfs.fasta"
    with open(output_file, "w") as f:
        SeqIO.write(orf_records, f, "fasta")


if __name__ == "__main__":
    # Get the input directory from the command line argument
    input_dir = sys.argv[1]

    # Process each GenBank file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".gb") or filename.endswith(".gbk"):
            gb_file = os.path.join(input_dir, filename)
            translate_orfs(gb_file)
