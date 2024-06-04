import argparse
import os
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq

# Parse command line arguments
parser = argparse.ArgumentParser(description="Annotate ORFs in intergenic regions of a GenBank file")
parser.add_argument("input_folder", help="Input folder with GenBank files")
parser.add_argument("-l","--orf_length", help="ORF length threshold", default=2)
args = parser.parse_args()
min_orf_length = args.orf_length
for filename in os.listdir(args.input_folder):
    if filename.endswith(".gb") or filename.endswith(".gbk"):
        input_file = os.path.join(args.input_folder, filename)
# Load the GenBank file
        record = SeqIO.read(input_file, "genbank")

# Get the intergenic regions as SeqFeatures
        intergenic_features = []
        orfs = []
        last_feature = None
        for feature in record.features:
            if feature.type == "CDS":
                if last_feature is not None and feature.location.start > last_feature.location.end:
                    intergenic_loc = FeatureLocation(last_feature.location.end + 1, feature.location.start)
                    intergenic_features.append(SeqFeature(intergenic_loc, type="intergenic"))
                last_feature = feature

        # Annotate ORFs and alternative start regions in the intergenic regions
        for intergenic_feature in intergenic_features:
            intergenic_seq = intergenic_feature.extract(record.seq)
            for strand, nuc in [(+1, intergenic_seq), (-1, intergenic_seq.reverse_complement())]:
                for frame in range(3):
                    trans = str(nuc[frame:].translate())
                    trans_len = len(trans)
                    aa_start = None
                    for aa_end in range(0, trans_len):
                        if trans[aa_end] == "M" or trans[aa_end] == "V":
                            aa_start = aa_end
                            if strand == 1:
                                orf_start = intergenic_feature.location.start + frame + aa_start * 3
                            if strand == -1:
                                orf_start = intergenic_feature.location.end - frame - aa_start * 3
                        elif aa_start is not None and trans[aa_end] == "*":
                            if strand == 1:
                                orf_end = intergenic_feature.location.start + frame + aa_end * 3 +2
                            if strand == -1:
                                orf_end = intergenic_feature.location.end - frame - aa_end * 3-2
                                [orf_end, orf_start] = [orf_start, orf_end]
                            orf_loc = FeatureLocation(orf_start, orf_end, strand=strand)
                            orf_feature = SeqFeature(orf_loc, type="ORF", qualifiers={
                                                    "translation": "M" + trans[aa_start+1:aa_end]})
                            if not any(f.type == "CDS" and orf_loc.start >= f.location.start and orf_loc.end <= f.location.end for f in record.features) and len(trans[aa_start:aa_end])>int(min_orf_length):
                                record.features.append(orf_feature)
                                orfs.append(orf_feature)
                            aa_start = None
                            print

# Write the updated GenBank file
        output_file = filename[:-3] + "_corefinder_output.gbk"
        SeqIO.write(record, output_file, "genbank")
        print(f"Output written to {output_file}")
        output_file = filename[:-3] + "_corefinder_output.fasta"
        with open(output_file, "w") as f:
            for orf in orfs:
                seq_id = f">{filename}|{orf.location.start}|{orf.location.end}|{orf.location.strand}"
                f.write(f"{seq_id}\n{orf.qualifiers['translation']}\n")

