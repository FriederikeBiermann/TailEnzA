#!/bin/bash
#SBATCH --job-name=20_cores_Muscle3_alignment_TerP450_%A_%a_
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -o /beegfs/projects/p450/TerP450_rerun/1NCBI_16_05_23_p450_prokaryotes_with_Reference/%A_%a_maxiters16_Muscle3_alignment.out
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-0%10
#SBATCH -e /beegfs/projects/p450/error_files/%A_%a_AA_maxiters16_Muscle3_alignment_single_enzyme_files.txt

# FirstRunMarker: YES

source ~/.bashrc
conda activate /beegfs/home/fbiermann/miniconda3_supernew/envs/muscle_38

input_file=$1  # Input file passed as an argument to the shell script
reference_file=$2  # Reference file passed as a second argument to the shell script
gap_open=$3  # Gap open penalty
gap_extend=$4  # Gap extension penalty

num_sequences=$(grep -c "^>" $input_file)
num_alignments=$((($num_sequences + 99) / 100))
num_jobs=$((($num_alignments + 9) / 10))

if [ $num_jobs -gt 1 ] && grep -q "FirstRunMarker: YES" $0; then
    sed -i "/FirstRunMarker: YES/c\# FirstRunMarker: NO" $0
    sed -i "/#SBATCH --array=0-0/c#SBATCH --array=0-$(($num_jobs - 1))%10" $0
    sbatch $0 $input_file $reference_file $gap_open $gap_extend
    exit 0
fi


for i in $(seq 0 9); do
    alignment_id=$(($SLURM_ARRAY_TASK_ID * 10 + $i))
    start=$(($alignment_id * 100))
    end=$(($start + 100))
    
    if [ $start -lt $num_sequences ]; then
        output_file="${input_file%.*}_aligned_$alignment_id.fasta"
        
        python3 run_muscle_3_with_parameters.py $input_file $output_file $reference_file $start $end $gap_open $gap_extend
    fi
done 
