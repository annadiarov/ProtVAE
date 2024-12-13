"""
Constants used in the project.
"""

# Available amino acids and padding token
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
PAD_TOKEN = '-'  # Padding token
TOKENS = f'{AMINO_ACIDS}{PAD_TOKEN}'

# VAE model architecture parameters
max_length = 164
input_dim = max_length * len(TOKENS)  # 20 amino acids + 1 padding
latent_dim = 128
hidden_dim = 256
