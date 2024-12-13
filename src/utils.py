"""
Utility functions for the project.
"""

import numpy as np
from constants import TOKENS, PAD_TOKEN

def one_hot_encode_sequence(
        sequence: str,
        max_length:int = 164
) -> np.ndarray:
    """One-hot encode a protein sequence"""
    sequence = sequence.upper()
    sequence = sequence + PAD_TOKEN * (max_length - len(sequence))
    one_hot = np.zeros((max_length, len(TOKENS)))
    for i, aa in enumerate(sequence):
        one_hot[i, TOKENS.index(aa)] = 1.
    return one_hot

def decode_one_hot_sequence(one_hot_sequence: np.ndarray) -> str:
    """Decode a one-hot encoded to protein sequence"""
    return "".join([TOKENS[np.argmax(i)] for i in one_hot_sequence])
