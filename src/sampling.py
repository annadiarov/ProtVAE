"""
Generate sequences with a trained VAE
"""

import os
import torch
import argparse as ap
from vae_model import VAE
from constants import max_length, input_dim, latent_dim, hidden_dim
from utils import decode_one_hot_sequence, PAD_TOKEN

def parse_args():
    parser = ap.ArgumentParser(description="Generate sequences with a trained VAE.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained VAE weights."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sequences to generate."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="generated_sequences.fasta",
        help="Path to the output file."
    )
    return parser.parse_args()

def main(
        weights_path: str,
        num_samples: int,
        output_file: str
):
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    vae.load_state_dict(torch.load(weights_path))
    vae.eval()

    with open(output_file, 'w') as f:
        for i in range(num_samples):
            z = torch.randn(1, latent_dim)
            generated_sequence = vae.decode(z).squeeze().detach().numpy()
            # Reshape from (max_length * 21) to (max_length, 21) and decode
            aa_sequence = decode_one_hot_sequence(
                generated_sequence.reshape(max_length, -1))
            # Remove padding tokens
            aa_sequence = aa_sequence.replace(PAD_TOKEN, "")
            f.write(f">Gen{i}\n{aa_sequence}\n")

    abs_output_file = os.path.abspath(output_file)
    print(f"Generated sequences saved to {abs_output_file}")

if __name__ == '__main__':
    args = parse_args()
    weights_path = args.weights
    num_samples = args.num_samples
    output_file = args.output_file
    main(weights_path, num_samples, output_file)
