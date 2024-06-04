from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import argparse
import pyhmmer

# Create Argument Parser for command line arguments
parser = argparse.ArgumentParser(description="Sequence extraction and HMMER search.")
parser.add_argument('input_path', help='Path to the dataset directory')
parser.add_argument('search_range_before', type=int, help='Number of CDS to check before the biosynthetic gene')
parser.add_argument('search_range_after', type=int, help='Number of CDS to check after the biosynthetic gene')
parser.add_argument('pfam_hmm_database', help='Path to PFAM HMM database')
parser.add_argument('output_path', help='Path to save HMMER search results')

args = parser.parse_args()

# Define BGC_types, gene_kind, and enzymes
#BGC_types = ["NRPS", "PKS", "Terpene", "RiPP"]
BGC_types = ["PKS", "Terpene", "RiPPs", "Alkaloide", "NRPSs"]
gene_kind = "biosynthetic"
enzymes = ["ycao", "P450", "radical SAM", "Methyltransf_2", "TP_methylase", "Methyltranf_3", "Methyltransf_25"]  # Replace with your list of enzymes

# Extraction of Sequences
for BGC_type in BGC_types:
    all_sequences = []
    bgc_dir = os.path.join(args.input_path, BGC_type)
    
    if not os.path.exists(bgc_dir) or not os.path.isdir(bgc_dir):
        print(f"Directory for enzyme '{BGC_type}' not found.")
        continue
    for root, _, files in os.walk(bgc_dir):
        for progenome_file in files:
            if progenome_file.endswith(".gbk"):
                for seq_record in SeqIO.parse(os.path.join(root, progenome_file), "gb"):
                    try:
                        enzyme_sequences = []
                        cds_features = [feature for feature in seq_record.features if feature.type == "CDS"]

                        # Iterate over the CDS features
                        for i, seq_feature in enumerate(cds_features):
                            # Check if the gene_kind is "biosynthetic"
                            if seq_feature.qualifiers.get("gene_kind", [""])[0] == gene_kind:
                                # Check surrounding CDS within the specified range
                                start_index = max(i - args.search_range_before, 0)
                                end_index = min(i + args.search_range_after + 1, len(cds_features))

                                # Iterate over nearby CDS features
                                for nearby_feature in cds_features[start_index:end_index]:
                                    if nearby_feature.qualifiers.get("gene_kind", [""])[0] != gene_kind:
                                        assert len(nearby_feature.qualifiers['translation']) == 1
                                        # Generate the enzyme sequence entry
                                        enzyme_sequence = SeqRecord(
                                                    Seq(nearby_feature.qualifiers['translation'][0]),  # sequence
                                                    id=nearby_feature.qualifiers['locus_tag'][0],  # ID
                                                    name=seq_record.name,  # Name
                                                    description=nearby_feature.qualifiers.get("product","No product predicted")[0]  # Description
                                                )
                                        enzyme_sequences.append(enzyme_sequence)

                        if enzyme_sequences:
                            all_sequences.extend(enzyme_sequences)
                    except Exception as e:
                        print("Error:", e)
    SeqIO.write(all_sequences, "temp.fasta", "fasta")
    # HMMER Search on Extracted Sequences
    for enzyme in enzymes:
        for filename in os.listdir(args.pfam_hmm_database):
            if f"{enzyme}.hmm" == filename:
                file_path = os.path.join(args.pfam_hmm_database, filename)
                output_file = os.path.join(args.output_path, f"{BGC_type}_{enzyme}_hmmer.fasta")
                
                sequences_dict = {seq.id: seq for seq in all_sequences}
                with pyhmmer.easel.SequenceFile("temp.fasta", digital=True) as seq_file:
                    sequences = seq_file.read_block()
                    with open(output_file, "w") as output_fasta_file:
                        with pyhmmer.plan7.HMMFile(file_path) as hmm_file:
                            for hits in pyhmmer.hmmsearch(hmm_file, sequences, cpus=4):
                                for hit in hits:
                                    evalue = hit.evalue
                                    hit_name = hit.name.decode()
                                    hit_description = hit.description.decode()
                                    extracted_part = f"{hit_name} {hit_description}"
                                    
                                    if evalue >= 10e-20:
                                        continue
                                    record = sequences_dict.get(hit_name)
                                    if record:
                                        SeqIO.write(record, output_fasta_file, "fasta")
                    print(f"Hits saved to {output_file}")
