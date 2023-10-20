import os
import sys
import subprocess
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment, AlignInfo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def run_muscle(input_file, output_file, gap_open=None, gap_extend=None, center=None):
    muscle_cmd = ["muscle", "-in", input_file, "-out", output_file , "-seqtype", "protein", "-maxiters", "16"]
    
    if gap_open:
        muscle_cmd.extend(["-gapopen", str(gap_open)])
    if gap_extend:
        muscle_cmd.extend(["-gapextend", str(gap_extend)])
    if center:
        muscle_cmd.extend(["-center", str(center)])

    
    try:
        subprocess.check_call(muscle_cmd)
    except subprocess.CalledProcessError:
        print(f"Error: Failed to run command {' '.join(muscle_cmd)}")
        sys.exit(1)

def adjust_alignments_using_msa(input_dir, reference_id="Reference"):
    alignment_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.fasta')]

    # Load your multiple sequence alignments
    alignments = [AlignIO.read(file, "fasta") for file in alignment_files]

    # Extract reference sequences
    references = [next(rec for rec in align if rec.id == reference_id) for align in alignments]

    # Replace '-' with 'X' in reference sequences and write them to a temporary file
    temp_file = "temp_references.fasta"
    with open(temp_file, "w") as f:
        for ref in references:
            f.write(f">{ref.id}\n{str(ref.seq).replace('-', 'X')}\n")

    # Align using MUSCLE
    output_file = "muscle_aligned_references.fasta"
    run_muscle(temp_file, output_file)

    # Read the aligned references and replace 'X' back to '-'
    aligned_references = AlignIO.read(output_file, "fasta")

    # Adjust the alignments based on the MSA of the reference sequences
    for idx, alignment in enumerate(alignments):
        adjusted_alignment = _adjust_alignment_based_on_reference(alignment, references[idx], aligned_references[idx].seq)

        # Save each adjusted alignment with a derived name from their original filename
        adjusted_filename = f"adjusted_{os.path.basename(alignment_files[idx])}"
        with open(adjusted_filename, "w") as output_file:
            AlignIO.write(adjusted_alignment, output_file, "fasta")

    # Optionally delete the temporary files
    os.remove(temp_file)
    os.remove(output_file)



def _adjust_alignment_based_on_reference(alignment, reference, aligned_reference):
    aligned_ref_str = str(aligned_reference)
    ref_str = str(reference.seq)
    
    # Mapping of position in aligned reference to position in original reference
    aligned_to_original = {}
    aligned_pos, original_pos = 0, 0
    while aligned_pos < len(aligned_ref_str) and original_pos < len(ref_str):
        if aligned_ref_str[aligned_pos] != '-':
            if ref_str[original_pos] != '-':
                aligned_to_original[aligned_pos] = original_pos
                original_pos += 1
            else:
                original_pos += 1
                continue
        aligned_pos += 1
    
    adjusted_records = []
    for record in alignment:
        adjusted_seq = []
        original_seq_str = str(record.seq)
        for i in range(len(aligned_ref_str)):
            if i in aligned_to_original:
                adjusted_seq.append(original_seq_str[aligned_to_original[i]])
            else:
                adjusted_seq.append('-')
        record.seq = Seq("".join(adjusted_seq))
        adjusted_records.append(record)

    return MultipleSeqAlignment(adjusted_records)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_directory>")
        sys.exit(1)
    input_directory = sys.argv[1]
    adjust_alignments_using_msa(input_directory)
