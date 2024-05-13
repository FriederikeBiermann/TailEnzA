import os
import re
import pandas as pd
from Bio import SeqIO

# Adjusted from https://github.com/nf-core/funcscan/blob/master/bin/comBGC.py
"""
===============================================================================
MIT License
===============================================================================

Copyright (c) 2023 Jasmin Frangenberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
verbose = True


def parse_clusterblast(cb_file_path):
    """
    Extract MIBiG ID and additional details of the best hit (first hit) from a knownclusterblast TXT file.
    """

    with open(cb_file_path) as cb_file:
        hits = 0
        GENOME_ID = ""
        additional_details = {}
        gene_count = -4
        gene_marker = False

        for line in cb_file:
            if line.startswith(
                "Table of genes, locations, strands and annotations of query cluster:"
            ):
                gene_marker = True
            if gene_marker == True:
                gene_count += 1
            if line.startswith("Significant hits:"):
                gene_marker = False
            if line.startswith("Details:"):
                gene_marker = False
                hits = 1  # Indicating that the following lines contain relevant information
            elif hits:
                if line.strip() and not line.startswith(">>"):
                    if "1. " in line:
                        GENOME_ID = re.search("1. (.+)", line).group(1)
                    elif "Source:" in line:
                        source = re.search("Source: (.+)", line).group(1)
                        additional_details["source"] = source
                    elif "Type:" in line:
                        type_info = re.search("Type: (.+)", line).group(1)
                        additional_details["type"] = type_info

                    elif "Number of proteins with BLAST hits to this cluster:" in line:
                        blast_gene = re.search(
                            "Number of proteins with BLAST hits to this cluster: (.+)",
                            line,
                        ).group(1)
                        additional_details["Percentage_genes_significant_hits"] = (
                            int(blast_gene) / gene_count
                        )
                    elif "Cumulative BLAST score:" in line:
                        blast_score = re.search(
                            "Cumulative BLAST score: (.+)", line
                        ).group(1)
                        additional_details["cumulative_blast_score"] = blast_score
                        break
    print(additional_details)
    return GENOME_ID, additional_details


def antismash_workflow(antismash_paths):
    """
    Create data frame with aggregated antiSMASH output:
    - Open summary GBK and grab relevant information.
    - Extract the knownclusterblast output from the antiSMASH folder (MIBiG annotations) if present.
    - Return data frame with aggregated info.
    """

    antismash_sum_cols = [
        "Sample_ID",
        "Prediction_tool",
        "Contig_ID",
        "Product_class",
        "BGC_probability",
        "BGC_complete",
        "BGC_start",
        "BGC_end",
        "BGC_length",
        "CDS_ID",
        "CDS_count",
        "PFAM_domains",
        "MIBiG_ID",
        "InterPro_ID",
    ]
    antismash_out = pd.DataFrame(columns=antismash_sum_cols)

    CDS_ID = []
    CDS_count = 0

    # Distinguish input files (i.e. GBK file and "knownclusterblast" folder)
    kcb_path = []
    cb_path = []
    for path in antismash_paths:
        if re.search("knownclusterblast", path):
            kcb_path = re.search(".*knownclusterblast.*", path).group()
        elif re.search("clusterblast", path):
            cb_path = re.search(".*clusterblast.*", path).group()
        else:
            gbk_path = path

    kcb_files = []
    if kcb_path:
        kcb_files = [file for file in os.listdir(kcb_path) if file.endswith(".txt")]

    # Distinguish input files (i.e. GBK file and "clusterblast" folder)

    cb_files = []
    if cb_path:
        cb_files = [file for file in os.listdir(cb_path) if file.endswith(".txt")]

    # Aggregate information
    Sample_ID = gbk_path.split("/")[-1].split(".gbk")[
        -2
    ]  # Assuming file name equals sample name
    if verbose:
        print("\nParsing antiSMASH file(s): " + Sample_ID + "\n... ", end="")

    with open(gbk_path) as gbk:
        for record in SeqIO.parse(
            gbk, "genbank"
        ):  # GBK records are contigs in this case
            # Initiate variables per contig
            cluster_num = 1
            antismash_out_line = {}
            Contig_ID = record.id
            Product_class = ""
            BGC_complete = ""
            BGC_start = ""
            BGC_end = ""
            BGC_length = ""
            PFAM_domains = []
            MIBiG_ID = "NA"
            additional_mibig_details = "NA"
            additional_refseq_details = "NA"
            RefSeq = "NA"

            for feature in record.features:
                # Extract relevant infos from the first protocluster feature from the contig record
                if feature.type == "protocluster":
                    if (
                        antismash_out_line
                    ):  # If there is more than 1 BGC per contig, reset the output line for new BGC. Assuming that BGCs do not overlap.
                        if not CDS_ID:
                            CDS_ID = ["NA"]
                        antismash_out_line = {  # Create dictionary of BGC info
                            "Sample_ID": Sample_ID,
                            "Prediction_tool": "antiSMASH",
                            "Contig_ID": Contig_ID,
                            "Product_class": ";".join(Product_class),
                            "BGC_probability": "NA",
                            "BGC_complete": BGC_complete,
                            "BGC_start": BGC_start,
                            "BGC_end": BGC_end,
                            "BGC_length": BGC_length,
                            "CDS_ID": ";".join(CDS_ID),
                            "CDS_count": CDS_count,
                            "PFAM_domains": ";".join(PFAM_domains),
                            "MIBiG_ID": MIBiG_ID,
                            "InterPro_ID": "NA",
                        }
                        antismash_out_line = pd.DataFrame([antismash_out_line])
                        antismash_out = pd.concat(
                            [antismash_out, antismash_out_line], ignore_index=True
                        )
                        antismash_out_line = {}

                        # Reset variables per BGC
                        CDS_ID = []
                        CDS_count = 0
                        PFAM_domains = []

                    # Extract all the BGC info
                    Product_class = feature.qualifiers["product"]
                    for i in range(len(Product_class)):
                        Product_class[i] = (
                            Product_class[i][0].upper() + Product_class[i][1:]
                        )  # Make first letters uppercase, e.g. lassopeptide -> Lassopeptide

                    if feature.qualifiers["contig_edge"] == ["True"]:
                        BGC_complete = "No"
                    elif feature.qualifiers["contig_edge"] == ["False"]:
                        BGC_complete = "Yes"

                    BGC_start = (
                        feature.location.start + 1
                    )  # +1 because zero-based start position
                    BGC_end = feature.location.end
                    BGC_length = feature.location.end - feature.location.start + 1
                    # If there are knownclusterblast files for the BGC, get MIBiG IDs of their homologs
                    if kcb_files:
                        kcb_file = "{}_c{}.txt".format(
                            record.id, str(cluster_num)
                        )  # Check if this filename is among the knownclusterblast files
                        if kcb_file in kcb_files:
                            MIBiG_ID, additional_mibig_details = parse_clusterblast(
                                os.path.join(kcb_path, kcb_file)
                            )
                    if cb_files:
                        print(cb_path, cb_files)
                        cb_file = "{}_c{}.txt".format(
                            record.id, str(cluster_num)
                        )  # Check if this filename is among the knownclusterblast files
                        if cb_file in cb_files:
                            RefSeq, additional_refseq_details = parse_clusterblast(
                                os.path.join(cb_path, cb_file)
                            )

                            cluster_num += 1

                # Count functional CDSs (no pseudogenes) and get the PFAM annotation
                elif (
                    feature.type == "CDS"
                    and "translation" in feature.qualifiers.keys()
                    and BGC_start != ""
                ):  # Make sure not to count pseudogenes (which would have no "translation tag") and count no CDSs before first BGC
                    if (
                        feature.location.end <= BGC_end
                    ):  # Make sure CDS is within the current BGC region
                        if "locus_tag" in feature.qualifiers:
                            CDS_ID.append(feature.qualifiers["locus_tag"][0])
                        CDS_count += 1
                        if "sec_met_domain" in feature.qualifiers.keys():
                            for PFAM_domain in feature.qualifiers["sec_met_domain"]:
                                PFAM_domain_name = re.search(
                                    "(.+) \(E-value", PFAM_domain
                                ).group(1)
                                PFAM_domains.append(PFAM_domain_name)

            # Create dictionary of BGC info
            if not CDS_ID:
                CDS_ID = ["NA"]
            antismash_out_line = {
                "Sample_ID": Sample_ID,
                "Prediction_tool": "antiSMASH",
                "Contig_ID": Contig_ID,
                "Product_class": ";".join(Product_class),
                "BGC_probability": "NA",
                "BGC_complete": BGC_complete,
                "BGC_start": BGC_start,
                "BGC_end": BGC_end,
                "BGC_length": BGC_length,
                "CDS_ID": ";".join(CDS_ID),
                "CDS_count": CDS_count,
                "PFAM_domains": ";".join(PFAM_domains),
                "MIBiG_ID": MIBiG_ID,
                "MIBiG_details": additional_mibig_details,
                "RefSeq": RefSeq,
                "RefSeq_details": additional_refseq_details,
                "InterPro_ID": "NA",
            }

            if BGC_start != "":  # Only keep records with BGCs
                antismash_out_line = pd.DataFrame([antismash_out_line])
                antismash_out = pd.concat(
                    [antismash_out, antismash_out_line], ignore_index=True
                )

                # Reset variables per BGC
                CDS_ID = []
                CDS_count = 0
                PFAM_domains = []

    return antismash_out


def process_antismash_directory(directory):
    all_antismash_data = pd.DataFrame()

    for subdir, dirs, files in os.walk(directory):
        gbk_files = [
            os.path.join(subdir, file)
            for file in files
            if (file.endswith(".gbk") and "region" not in file)
        ]
        cb_dirs = [os.path.join(subdir, dir) for dir in dirs if "clusterblast" in dir]

        for gbk_file in gbk_files:
            antismash_paths = [gbk_file] + cb_dirs
            antismash_data = antismash_workflow(antismash_paths)
            all_antismash_data = pd.concat(
                [all_antismash_data, antismash_data], ignore_index=True
            )

    return all_antismash_data


main_directory = "/projects/p450/TailEnzA_Interference_NCBI_21_11_2023/output/antiSMASH_output/over_1/"
final_dataframe = process_antismash_directory(main_directory)
final_dataframe.to_csv(
    f"{os.path.basename(main_directory)}_antismash_output_analysis.csv", index=False
)
