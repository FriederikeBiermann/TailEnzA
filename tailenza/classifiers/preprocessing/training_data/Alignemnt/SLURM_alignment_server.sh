#!/bin/bash 
#SBATCH --job-name=Muscle_alignment
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH -o /beegfs/projects/p450/Muscle_alignment.out  
#SBATCH -t 1-00:00  
#SBATCH -p batch 
#SBATCH -e /beegfs/projects/p450/Muscle_alignment_error.txt



source ~/.bashrc
conda activate /beegfs/home/fbiermann/miniconda3/envs/antismash_new

muscle -in NRPS_antismash_DB_radical_SAM_all_output_new.fasta -out NRPS_antismash_DB_radical_SAM_all_output_new_aligned.fasta -matrix matrix_first_try.txt -gapopen -2.0 -gapextend -1.0 -center 0.0  -seqtype protein
