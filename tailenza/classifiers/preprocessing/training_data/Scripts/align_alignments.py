import os
import sys
import subprocess
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment, AlignInfo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def run_muscle(input_file, output_file, gap_open=0.1, gap_extend=0.1, center=0.1):
    muscle_cmd = [
        "muscle",
        "-in",
        input_file,
        "-out",
        output_file,
        "-seqtype",
        "protein",
        "-maxiters",
        "16",
    ]

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
    alignment_files = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".fasta")
    ]

    # Load your multiple sequence alignments
    alignments = []
    for file in alignment_files:
        alignment = AlignIO.read(file, "fasta")
        adjusted_alignment = []

        for record in alignment:
            adjusted_seq = str(record.seq).replace("-", "X")
            new_record = SeqRecord(
                Seq(adjusted_seq), id=record.id, description=record.description
            )
            adjusted_alignment.append(new_record)

        alignments.append(adjusted_alignment)
    # Extract reference sequences
    references = [
        next(
            SeqRecord(rec.seq, id=f"{rec.id}_{index}")
            for rec in align
            if rec.id == reference_id
        )
        for index, align in enumerate(alignments)
    ]
    print(references)
    temp_file = f"{input_dir}_temp_references.fasta"
    with open(temp_file, "w") as f:
        for index, ref in enumerate(references):
            f.write(f">{ref.id}\n{str(ref.seq)}\n")

    # Align using MUSCLE
    output_file = f"{input_dir}_muscle_aligned_references.fasta"
    run_muscle(temp_file, output_file)

    aligned_references = AlignIO.read(output_file, "fasta")
    # Reorder alignment

    indices = [int(record.id.split("_")[-1]) for record in aligned_references]

    # Create a dictionary mapping the index to the record
    index_to_record = {
        int(record.id.split("_")[-1]): record for record in aligned_references
    }

    # Reorder the alignment based on the indices
    reordered_alignment = [index_to_record[index] for index in sorted(indices)]

    # Adjust the alignments based on the MSA of the reference sequences
    print(references)
    for idx, alignment in enumerate(alignments):
        adjusted_alignment = _adjust_alignment_based_on_reference(
            alignment, references[idx], reordered_alignment[idx].seq
        )
        final_adjusted_alignment = []
        for record in adjusted_alignment:
            final_seq = str(record.seq).replace("X", "-")
            final_record = SeqRecord(
                Seq(final_seq), id=record.id, description=record.description
            )
            final_adjusted_alignment.append(final_record)
        # Save each adjusted alignment with a derived name from their original filename
        adjusted_filename = f"adjusted_{os.path.basename(alignment_files[idx])}"
        with open(adjusted_filename, "w") as output_file:
            AlignIO.write(
                MultipleSeqAlignment(final_adjusted_alignment), output_file, "fasta"
            )

    # Optionally delete the temporary files
    # os.remove(temp_file)


def _adjust_alignment_based_on_reference(alignment, reference, aligned_reference):
    aligned_ref_str = str(aligned_reference)
    ref_str = str(reference.seq)
    print(aligned_ref_str, ref_str)
    # Mapping of position in aligned reference to position in original reference
    aligned_to_original = {}
    aligned_pos, original_pos = 0, 0
    while aligned_pos < len(aligned_ref_str):
        if aligned_ref_str[aligned_pos] != "-":
            if ref_str[original_pos] != "-":
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
                adjusted_seq.append("-")
        record.seq = Seq("".join(adjusted_seq))
        adjusted_records.append(record)

    return MultipleSeqAlignment(adjusted_records)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_directory>")
        sys.exit(1)
    input_directory = sys.argv[1]
    adjust_alignments_using_msa(input_directory)
