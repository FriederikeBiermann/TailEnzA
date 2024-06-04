import os
import sys
import subprocess
from Bio import SeqIO

def run_muscle(input_file, output_file, gap_open=None, gap_extend=None, center=None):
    muscle_cmd = ["muscle", "-in", input_file, "-out", output_file , "-seqtype", "protein", "-maxiters", "16"]
    
    if gap_open:
        muscle_cmd.extend(["-gapopen", str(gap_open)])
    if gap_extend:
        muscle_cmd.extend(["-gapextend", str(gap_extend)])
    if center:
        muscle_cmd.extend(["-center", str(center)])

    
   # try:
    subprocess.check_call(muscle_cmd)
    #except subprocess.CalledProcessError:
     #   print(f"Error: Failed to run command {' '.join(muscle_cmd)}")
     #   sys.exit(1)

try:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    reference_file = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    gap_open = float(sys.argv[6]) if len(sys.argv) > 6 else None
    gap_extend = float(sys.argv[7]) if len(sys.argv) > 7 else None
    center = float(sys.argv[8]) if len(sys.argv) > 8 else None 
except (IndexError, ValueError):
    print("Usage: python script.py <input_file> <output_file> <reference_file> <start> <end> [<gap_open> <gap_extend> <center>]")
    sys.exit(1)

# Read Reference Sequence
try:
    with open(reference_file, "r") as f:
        reference_seq = next(SeqIO.parse(f, "fasta"))
    reference_seq.id = "Reference"
except (FileNotFoundError, StopIteration):
    print(f"Error: The file {reference_file} was not found or is empty.")
    sys.exit(1)

# Read a subset of Input Sequences based on start and end
try:
    with open(input_file, "r") as f:
        sequences = list(SeqIO.parse(f, "fasta"))[start:end]
    if not sequences:
        raise ValueError(f"No sequences found in {input_file} between {start} and {end}")
except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
    sys.exit(1)
except ValueError as e:
    print(e)
    sys.exit(1)

# Create a Temporary File with Reference and Original Filename
original_filename = os.path.basename(output_file)
temp_file = f"temp_with_reference_{original_filename}"
with open(temp_file, "w") as f:
    SeqIO.write([reference_seq] + sequences, f, "fasta")

run_muscle(temp_file, output_file, gap_open, gap_extend, center)
