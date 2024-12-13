"""
This module contains the dataset class for the fasta dataset.
"""

import torch
import numpy as np
from Bio import SeqIO
from torch.utils.data import Dataset
from utils import one_hot_encode_sequence


class FastaDataset(Dataset):
    def __init__(self, fasta_file, max_length=164):
        self.sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            one_hot = one_hot_encode_sequence(str(record.seq), max_length)
            self.sequences.append(one_hot)
        self.sequences = np.array(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)
