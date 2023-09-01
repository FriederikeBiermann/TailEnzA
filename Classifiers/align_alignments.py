from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment, AlignInfo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def adjust_alignments_using_msa(*alignment_files, reference_id="Reference"):
    """
    Adjusts multiple sequence alignments using a multiple sequence alignment of their reference sequences.

    Parameters:
    *alignment_files: Paths to the alignment files to be adjusted.
    reference_id (str): ID of the reference sequence in the alignment.

    Returns:
    List of adjusted MultipleSeqAlignments.
    """

    # Load your multiple sequence alignments
    alignments = [AlignIO.read(file, "fasta") for file in alignment_files]

    # Extract reference sequences
    references = [next(rec for rec in align if rec.id == reference_id) for align in alignments]

    # Multiple Sequence Alignment on all reference sequences
    msa = AlignInfo.MultipleSeqAlignment(references)

    # Adjust the alignments based on the MSA of the reference sequences
    for idx, alignment in enumerate(alignments):
        alignments[idx] = _adjust_alignment_based_on_reference(alignment, references[idx], msa[idx].seq)

    return alignments

def _adjust_alignment_based_on_reference(alignment, reference, aligned_reference):
    """
    Adjusts an alignment based on the alignment of its reference sequence.

    Parameters:
    alignment (MultipleSeqAlignment): The alignment to be adjusted.
    reference (SeqRecord): The original reference sequence from the alignment.
    aligned_reference (Seq): The aligned reference sequence from the MSA.

    Returns:
    Adjusted MultipleSeqAlignment.
    """
    
    adjusted_records = []
    original_seq = str(reference.seq)
    index_original = 0
    index_aligned = 0

    while index_aligned < len(aligned_reference):
        # If the position in the aligned reference has a gap, insert a gap in all sequences
        if aligned_reference[index_aligned] == '-':
            for record in alignment:
                record.seq = record.seq[:index_original] + '-' + record.seq[index_original:]
            index_aligned += 1
        else:
            index_original += 1
            index_aligned += 1

    return alignment

# Example usage:
adjusted_alignments = adjust_alignments_using_msa("alignment1.fasta", "alignment2.fasta", "alignment3.fasta")
 
