from torch.utils.data import Dataset, DataLoader
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
import random
import torch
from torch.utils.data import DataLoader, Sampler

# the splitting list defines the functional fragments that the enzymes will be cut into
SPLITTING_SITES={
    'p450': {
        'begin': {'start': 1, 'end': 92},
        'sbr1': {'start': 93, 'end': 192},
        'sbr2': {'start': 193, 'end': 275},
        'core': {'start': 276, 'end': 395},
        'end': {'start': 396, 'end': None},
        'fes1': {'start': 54, 'end': 115},
        'fes2': {'start': 302, 'end': 401}
    },
    'ycao': {
        'begin': {'start': 1, 'end': 64},
        'sbr1': {'start': 65, 'end': 82},
        'f2': {'start': 83, 'end': 153},
        'sbr2': {'start': 154, 'end': 185},
        'f3': {'start': 186, 'end': 227},
        'sbr3': {'start': 228, 'end': 281},
        'f4': {'start': 282, 'end': 296},
        'sbr4': {'start': 297, 'end': 306},
        'f5': {'start': 307, 'end': 362},
        'sbr5': {'start': 363, 'end': 368},
        'end': {'start': 369, 'end': None}
    },
    'SAM': {
        'begin': {'start': 1, 'end': 106},
        'SAM': {'start': 107, 'end': 310},
        'bridging': {'start': 311, 'end': 346},
        'end': {'start': 347, 'end': None}
    },
    'Methyl': {
        'begin': {'start': 1, 'end': 78},
        'SAM1': {'start': 79, 'end': 104},
        'SAM2': {'start': 105, 'end': 128},
        'SAM3': {'start': 129, 'end': 158},
        'SAM4': {'start': 159, 'end': 188},
        'SAM5': {'start': 189, 'end': 233},
        'end': {'start': 234, 'end': None}
    }
}




class MultiDataLoaderSampler(Sampler):
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.num_samples = sum([len(dl.dataset) for dl in dataloaders])
    
    def __iter__(self):
        iterators = [iter(dl) for dl in self.dataloaders]
        indices = list(range(len(self.dataloaders)))
        random.shuffle(indices)
        while indices:
            random.shuffle(indices)
            idx = indices.pop()
            try:
                yield next(iterators[idx])
            except StopIteration:
                pass
                
    def __len__(self):
        return self.num_samples

class CustomDataset(Dataset):
    """
    Custom dataset class to handle sequence data for given enzymes and BGC types.
    """

    def __init__(self, foldername_training_sets, enzyme, BGC_type):
        """
        Initialize the dataset with the given folder name, enzyme, and BGC type.

        Parameters:
        - foldername_training_sets: Path to the folder containing the training datasets.
        - enzyme: Specified enzyme for dataset generation.
        - BGC_type: Specified BGC type for dataset generation.
        """
        # Load the data from the fasta file
        filename = foldername_training_sets + enzyme + "_" + BGC_type + ".fasta"
        self.data = AlignIO.read(open(filename), "fasta")
        self.bgc_type = BGC_type

        # Finding and indexing reference from the data
        self.reference_record = self.find_reference(self.data)
        self.index_reference = self.index_reference(self.reference_record)

        # Converting splitting list based on reference indexing
        self.splitting_list = self.convert_splitting_list(SPLITTING_SITES[enzyme], self.index_reference)
        self.fragments = self.splitting_list.keys()
        self.fragment_legths = [value.end - value.start for _, value in self.splitting_list.items()]

    def find_reference(self, alignment, name_reference="Reference"):
        """
        Find the reference record in the alignment.

        Parameters:
        - alignment: The sequence alignment.
        - name_reference (optional): The name of the reference sequence. Default is 'Reference'.

        Returns:
        - Reference sequence record.
        """
        for record in alignment:
            if record.id == name_reference:
                return record

    def index_reference(self, record):
        """
        Index the reference sequence, accommodating for gaps.

        Parameters:
        - record: The sequence record.

        Returns:
        - A list of mapped indices.
        """
        list_reference = list(str(record.seq))
        index_aa = 0
        index_mapping = []
        for index, AA in enumerate(list_reference):
            if AA != "-":
                index_aa += 1
                index_mapping.append([index_aa, index])
        return index_mapping

    def convert_splitting_list(self, splitting_list, index_reference):
        """
        Convert the canonical splitting list to also reflect eventual gaps in the reference sequence.

        Parameters:
        - splitting_list: The initial splitting list.
        - index_reference: The indexed reference mapping.

        Returns:
        - A dictionary with the converted splitting list.
        """
        converted_splitting_list = {}
        for fragment, fragment_rules in splitting_list.items():
            if fragment_rules.end:
                converted_splitting_list[fragment] = {
                    "start": index_reference[fragment_rules.start][1], 
                    "end": index_reference[fragment_rules.end][1]
                }
            else:
                converted_splitting_list[fragment] = {
                    "start": index_reference[fragment_rules.start][1], 
                    "end": index_reference[-1][1]
                }
                
        return converted_splitting_list

    def calculate_charge(self, sequence):
        """
        Calculate the approximate charge of an amino acid sequence.

        Parameters:
        - sequence: The amino acid sequence.

        Returns:
        - Calculated charge of the sequence.
        """
        AACharge = {"C": -.045, "D": -.999, "E": -.998, "H": .091, "K": 1, "R": 1, "Y": -.001}
        charge = -0.002
        for aa in sequence:
            charge += AACharge.get(aa, 0)  # get charge if aa exists in the dictionary, else add 0
        return charge

    def encode_sequence(self, sequence):
        """
        Directly encodes the amino acid sequence into a one-hot representation based on physico-chemical properties.

        Parameters:
        - sequence: The amino acid sequence.

        Returns:
        - A one-hot encoded representation of the sequence.
        """
        encoding = {
            # ... (similar to the provided encoding dictionary) ...
        }
        return [encoding[aa] for aa in sequence]

    def calculate_basic_percentage(self, sequence):
        """
        Calculate the percentage of basic amino acids in a given sequence.

        Parameters:
        - sequence: The amino acid sequence.

        Returns:
        - Percentage of basic amino acids in the sequence.
        """
        basic_amino_acids = ['R', 'H', 'K']
        count = sum([sequence.count(aa) for aa in basic_amino_acids])
        return count / len(sequence)

    def calculate_acidic_percentage(self, sequence):
        """
        Calculate the percentage of acidic amino acids in a given sequence.

        Parameters:
        - sequence: The amino acid sequence.

        Returns:
        - Percentage of acidic amino acids in the sequence.
        """
        acidic_amino_acids = ['D', 'E']
        count = sum([sequence.count(aa) for aa in acidic_amino_acids])
        return count / len(sequence)

    def __len__(self):
        """ Returns the total number of records in the dataset. """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetch an item from the dataset given an index.

        Parameters:
        - idx: Index of the desired item.

        Returns:
        - Tuple containing record id, encoded fragment, charge tensors, and sequence lengths.
        """
        record = self.data[idx]
        charge_tensors = []
        encoded_fragment = torch.tensor(self.encode_sequence(record.seq), dtype=torch.float32)

        for fragment_name in self.fragments:
            fragment_sequence = record.seq[self.splitting_list[fragment_name].start:self.splitting_list[fragment_name].end]

            # Calculate charges
            charge = torch.tensor(self.calculate_charge(fragment_sequence), dtype=torch.float32)
            charge_tensors.append(charge)
            
            # Calculate basic and acidic percentages for the fragment
            basic_percentage = torch.tensor(self.calculate_basic_percentage(fragment_sequence), dtype=torch.float32)
            charge_tensors.append(basic_percentage)

            acidic_percentage = torch.tensor(self.calculate_acidic_percentage(fragment_sequence), dtype=torch.float32)
            charge_tensors.append(acidic_percentage)

        # Add basic and acidic percentages for the entire sequence
        charge_tensors.append(torch.tensor(self.calculate_charge(record.seq), dtype=torch.float32))
        charge_tensors.append(torch.tensor(self.calculate_basic_percentage(record.seq), dtype=torch.float32))
        charge_tensors.append(torch.tensor(self.calculate_acidic_percentage(record.seq), dtype=torch.float32))
        
        charge_tensors = torch.tensor(charge_tensors, dtype=torch.float32)

        return record.id, encoded_fragment, charge_tensors, self.bgc_type


