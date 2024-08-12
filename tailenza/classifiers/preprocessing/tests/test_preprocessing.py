import pytest
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import pandas as pd
from importlib.resources import files
import torch
import esm
from tailenza.classifiers.preprocessing.feature_generation import AlignmentDataset
from pathlib import Path

# Mock P450 enzyme data
P450_ENZYME_DATA = {
    "P450": {
        "splitting_list": {
            "begin": [1, 69],
            "sbr1": [70, 78],
            "str1": [79, 195],
            "sbr2": [196, 228],
            "cat": [229, 235],
            "str2": [236, 377],
            "sbr3": [378, 383],
            "end": [384, 391],
        },
        "hmm_file": "p450.hmm",
        "classifier_BGC_type": "_AdvancedFFNN/AdvancedFFNN_model.pth",
        "classifier_metabolism": "_LSTM/LSTM_model.pth",
        "reference_for_alignment": "PPPSLEDAAPSVLRLSPLLRELQMRAPVTKIRTPAGDEGWLVTRHAELKQLLHDERLARAHADPANAPRYVKSPLMDLLIMDDVEAARAAHAELRTLLTPQFSARRVLNMMPMVEGIAEQILNGFAAQEQPADLRGNFSLPYSLTVLCALIGIPLQEQGQLLAVLGEMATLNDAESVARSQAKLFGLLTDLAGRKRAEPGDDVISRLCETVPEDERIGPIAASLLFAGLDSVATHVDLGVVLFTQYPDQLKEALADEKLMRSGVEEILRAAKAGGSGAALPRYATDDIEIADVTIRTGDLVLLDFTLVNFDEAVFDDADLFDIRRSPNEHLTFGHGMWHCIGAPLARMMLKTAYTQLFTRLPGLKLASSVEELQVTSGQLNGGLTELPVTW",
        "gap_opening_penalty": -2,
        "gap_extend_penalty": -1,
        "center": -1,
        "min_length": 200,
        "max_length": 800,
    }
}

# Load the ESM-1b model and batch converter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
package_dir = files("tailenza").joinpath("")
file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
model, alphabet = esm.pretrained.load_model_and_alphabet_local(file_path_model)
batch_converter = alphabet.get_batch_converter()
model = model.eval()


@pytest.fixture
def dataset():
    """Fixture to set up the AlignmentDataset object."""
    alignment_path = package_dir.joinpath(
        "classifiers",
        "preprocessing",
        "tests",
        "test_data",
        "test_pk_p450_alignment.fas",
    )
    alignment = AlignIO.read(alignment_path, "fasta")
    return AlignmentDataset(P450_ENZYME_DATA, "P450", alignment)


def test_filter_alignment(dataset):
    """Test filtering alignment by sequence length."""
    original_length = len(dataset.alignment)
    filtered_alignment = dataset._filter_alignment()
    assert len(filtered_alignment) >= 1  # At least one sequence should be included
    assert original_length - 3 == len(filtered_alignment)
    # One sequence should be too long, one should be too short


def test_fragment_alignment(dataset):
    """Test fragment alignment."""
    dataset._filter_alignment()
    fragment_matrix = dataset.fragment_alignment(fastas_aligned_before=True)
    assert "begin" in fragment_matrix.columns
    assert (
        fragment_matrix.shape[0] == len(dataset.alignment) - 1
    )  # The reference sequence is filtered out
    expected_columns = set(P450_ENZYME_DATA["P450"]["splitting_list"].keys())
    expected_columns.add("Concatenated")
    assert set(fragment_matrix.columns) == expected_columns


def test_featurize_fragments(dataset):
    """Test featurizing the fragments."""
    dataset._filter_alignment()
    dataset.fragment_alignment(fastas_aligned_before=True)
    feature_matrix = dataset.featurize_fragments(batch_converter, model)
    assert feature_matrix is not None
    assert (
        dataset.feature_matrix.shape[0] == len(dataset.alignment) - 1
    )  # The reference sequence is filtered out
    assert "charge_begin" in feature_matrix.columns
    assert "begin_0" in feature_matrix.columns


def test_convert_splitting_list(dataset):
    """Test converting the splitting list to account for reference sequence gaps."""
    dataset._filter_alignment()
    reference = SeqRecord(Seq(dataset.reference_sequence), id="Reference")
    index_reference = dataset._indexing_reference(reference)
    converted_splitting_list = dataset._convert_splitting_list(
        dataset.splitting_list, index_reference
    )
    assert len(converted_splitting_list) == len(dataset.splitting_list)
    assert converted_splitting_list[0][0] == 1
    assert converted_splitting_list != dataset.splitting_list


def test_calculate_charge(dataset):
    """Test calculating the charge of a sequence."""
    charge = dataset._calculate_charge("PPPSLEDAAPSVLRLSPLL")
    assert isinstance(charge, float)
    assert charge == -0.09990000000000002


def test_remove_incomplete_rows(dataset):
    """Test removing incomplete rows from the fragment matrix."""
    dataset._filter_alignment()
    dataset.fragment_alignment(fastas_aligned_before=True)
    initial_shape = dataset.fragment_matrix.shape
    dataset.fragment_matrix.iloc[0, 0] = ""  # Introduce an empty fragment
    cleaned_matrix = dataset._remove_incomplete_rows(dataset.fragment_matrix)
    assert cleaned_matrix.shape[0] == initial_shape[0] - 1  # One row should be removed


if __name__ == "__main__":
    pytest.main()
