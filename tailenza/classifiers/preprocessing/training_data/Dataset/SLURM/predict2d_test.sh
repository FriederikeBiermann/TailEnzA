#!/bin/bash 
#SBATCH --job-name=2dpred_test
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH -o /beegfs/projects/p450/out_files/testfile_2dpred.out  
#SBATCH -t 1-00:00  
#SBATCH -p batch 
#SBATCH -e /beegfs/projects/p450/error_files/testfile_2dpred.txt



source ~/.bashrc
conda activate /beegfs/home/fbiermann/miniconda3/envs/2dpred_new


python /beegfs/projects/p450/2dpred/predict2d.py -i /beegfs/projects/p450/2dpred/Testfile.fasta -o /beegfs/projects/p450/2dpred/output/ -t /beegfs/projects/p450/2dpred/tmp/ -c 2 --parallel 4 -p /beegfs/projects/p450/2dpred/cd ..
cd out:Porter5/Porter5.py
