#!/bin/bash 
#SBATCH --job-name=PKS_antismash
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH -o /beegfs/projects/p450/out_files/PKS_antismash_%A_%a.out  
#SBATCH -t 1-00:00  
#SBATCH -p batch 
#SBATCH --array=1-200
#SBATCH -e /beegfs/projects/p450/error_files/PKS_antismash_error_%A_%a.txt


source ~/.bashrc
conda activate /beegfs/home/fbiermann/miniconda3/envs/2dpred_new




for file in /beegfs/projects/p450/PKS_genbank_files_antismash_DB/PKS_$SLURM_ARRAY_TASK_ID/*.gb ; do python /beegfs/projects/p450/2dpred/predict2d.py -i $file -o /beegfs/projects/p450/2dpred/output/ -t /beegfs/projects/p450/2dpred/tmp/ -c 2 --parallel 4 -p /beegfs/projects/p450/2dpred/cd ..
cd out:Porter5/Porter5.py
