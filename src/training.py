"""
Implements training loop for the VAE model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse as ap
from dataset import FastaDataset
from vae_model import VAE
from constants import max_length, input_dim, latent_dim, hidden_dim


def parse_args():
    parser = ap.ArgumentParser(description="Train a VAE model on protein sequences.")
    parser.add_argument(
        "--fasta-file",
        type=str,
        required=True,
        help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=42
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--output-weights',
        type=str,
        default='vae_weights.pth',
    )
    return parser.parse_args()


def vae_loss_function(recon_x, x, mean, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_x, x.argmax(dim=-1))
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_div


def main(
        fasta_file: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        output_weights: str
):
    # Load the dataset
    dataset = FastaDataset(fasta_file, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.view(batch.size(0), -1)  # Flatten input
            optimizer.zero_grad()
            recon_x, mean, logvar = vae(batch)
            loss = vae_loss_function(recon_x, batch, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Save the model weights
    torch.save(vae.state_dict(), output_weights)
    abs_output_weights = os.path.realpath(output_weights)
    print(f"Model weights saved to {abs_output_weights}")


if __name__ == "__main__":
    args = parse_args()
    fasta_file = args.fasta_file
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    output_weights = args.output_weights
    main(
        fasta_file=fasta_file,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        output_weights=output_weights
    )
