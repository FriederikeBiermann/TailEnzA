#!/bin/bash
#SBATCH --job-name=SS8_maxiters16_Muscle_alignment_gomin1_gemin2_%A_%a_
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH -o /beegfs/projects/p450/out_files/%A_%a_SS8_maxiters8_Muscle_alignment_gomin1_gemin2.out
#SBATCH -t 15-00:00
#SBATCH -p long
#SBATCH --array=1-12
#SBATCH -e /beegfs/projects/p450/error_files/%A_%a_SS8_maxiters8_Muscle_alignment_gomin1_gemin2_error.txt


source ~/.bashrc
conda activate /beegfs/home/fbiermann/miniconda3_supernew/envs/muscle_38

for file in /beegfs/projects/p450/2dpred/SS8_muscle_alignment/all_alignments/$SLURM_ARRAY_TASK_ID*.fasta; do

$echo $file
        muscle -in $file -out $file.gomin1_gemin2.afa -matrix /projects/p450/2dpred/SS8_muscle_alignment/matrix_first_try.txt -gapopen -1.0 -gapextend -2.0 -center 0.0  -seqtype protein -maxiters 16;
done
