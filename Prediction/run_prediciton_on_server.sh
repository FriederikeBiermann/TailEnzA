#!/bin/bash
#SBATCH --job-name=TailEnzA_Run
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=/projects/p450/out_files/%A_%a.out
#SBATCH --error=/projects/p450/error_files/%A_%a.err
#SBATCH --array=0-95%95  # Adjust according to the number of files divided by 200
#SBATCH --time=24:00:00  # Adjust as needed
#SBATCH -p batch
#SBATCH --mem=20G        # Adjust as needed
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate /beegfs/home/fbiermann/miniconda3_supernew/envs/Noemi

# Your Python script path
PYTHON_SCRIPT="prediction_module_new.py"

# Input directory containing subdirectories with files
INPUT_DIR="/projects/p450/NCBI_xanthomonas_norcardiaceae_myxococcales_actinomycetes_pseudomonadota/ncbi_dataset/output"

# Create a temporary file list
FILE_LIST=$(mktemp)
find "${INPUT_DIR}" -type f > "${FILE_LIST}"

# Calculate the total number of files
NUM_FILES=$(wc -l < "${FILE_LIST}")

# Calculate start and end line for this array job
START_LINE=$(( SLURM_ARRAY_TASK_ID * 200 + 1 ))
END_LINE=$(( START_LINE + 199 ))

if [ ${END_LINE} -gt ${NUM_FILES} ]; then
    END_LINE=${NUM_FILES}
fi

# Iterate over the range of lines for this job
for LINE in $(seq ${START_LINE} ${END_LINE}); do
    FILE=$(sed -n "${LINE}p" "${FILE_LIST}")
    
    # Get subdirectory and file name to create output directory
    SUB_DIR=$(dirname "${FILE}")
    SUB_DIR_NAME=$(basename "${SUB_DIR}")
    FILE_NAME=$(basename "${FILE}")

    # Create output directory
    OUTPUT_DIR="/path/to/output_directory/${SUB_DIR_NAME}_${FILE_NAME}"

    # Run your Python script with the necessary arguments
    python "${PYTHON_SCRIPT}" -i "${FILE}" -o "${OUTPUT_DIR}" -f 15000 -t 5000 -c 1
done

# Clean up
rm "${FILE_LIST}"
 
