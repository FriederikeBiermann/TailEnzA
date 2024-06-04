#!/bin/bash

# Check if an input directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

# Input directory
input_dir="$1"

# Define arrays for enzymes and BGC types
enzymes=("Methyltransf_2" "Methyltransf_3" "Methyltransf_25" "P450" "radical_SAM" "ycao" "TP_methylase")
BGC_types=("PKS" "Terpene" "Alkaloide" "NRPSs" "RiPPs")

# Reference file directory
ref_dir="/projects/p450/Training_data_Tailenza_18_11_2023_hmmer_4_genes_from_biosynthetic_without_hybrids/reference"

# Loop through each combination of enzyme and BGC type
for enzyme in "${enzymes[@]}"; do
    for BGC_type in "${BGC_types[@]}"; do
        # Set parameters and reference file based on enzyme type
        case "$enzyme" in
            "P450")
                gap_open=-2
                gap_extend=-1
                center=-1
                reference_file="${ref_dir}/P450_reference.fasta"
                ;;
            Methyltransf_2|Methyltransf_3|Methyltransf_25|TP_methylase)
                gap_open=-1
                gap_extend=-2
                center=0
                reference_file="${ref_dir}/${enzyme}.fasta"
                ;;
            "radical_SAM")
                gap_open=-2
                gap_extend=-1
                center=-2
                reference_file="${ref_dir}/SAM_rcsb_pdb_5V1T_reference.fasta"
                ;;
            "ycao")
                gap_open=-2
                gap_extend=-1
                center=-5
                reference_file="${ref_dir}/YCOA_reference.fasta"
                ;;
        esac

        # Construct the input file name
        input_file="${input_dir}/${BGC_type}_${enzyme}_hmmer_deduplicated.fasta"

        # Check if the input file exists
        if [ ! -f "$input_file" ]; then
            echo "Input file not found: $input_file"
            continue
        fi

        # Call the script with the constructed arguments
        sbatch run_muscle_3_batch_with_parameters.sh "$input_file" "$reference_file" "$gap_open" "$gap_extend" "$center"
    done
done

