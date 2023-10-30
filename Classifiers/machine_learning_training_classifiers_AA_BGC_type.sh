#!/bin/bash
#SBATCH --job-name=20_cores_Muscle3_alignment_TerP450_%A_%a_
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -o /beegfs/projects/p450/out_files/%A_%a_maxiters16_Muscle3_alignment.out
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH -e /beegfs/projects/p450/error_files/%A_%a_AA_maxiters16_Muscle3_alignment_single_enzyme_files.txt

source ~/.bashrc
conda activate /beegfs/home/fbiermann/miniconda3_supernew/envs/Noemi

python  machine_learning_training_classifiers_AA_BGC_type.py
