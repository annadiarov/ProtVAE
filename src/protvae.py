import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Bio import SeqIO

# Input parameters
fasta_file = "data/example.fasta"  # Replace with your FASTA file
max_length = 164                   # Maximum sequence length
batch_size = 32                    # Batch size for training

# Constants
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
PAD_TOKEN = '-'  # Padding token
TOKENS = f'{AMINO_ACIDS}{PAD_TOKEN}'

# Define a function to one-hot encode amino acid sequences
def one_hot_encode_sequence(sequence, max_length=164):
    sequence = sequence.upper()
    sequence = sequence + PAD_TOKEN * (max_length - len(sequence))
    one_hot = np.zeros((max_length, len(TOKENS)))
    for i, aa in enumerate(sequence):
        one_hot[i, TOKENS.index(aa)] = 1.
    return one_hot

def decode_one_hot_sequence(one_hot_sequence):
    return "".join([TOKENS[np.argmax(i)] for i in one_hot_sequence])

# Define a custom PyTorch Dataset class for FASTA data
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


dataset = FastaDataset(fasta_file, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the VAE class
class VAE(nn.Module):
    def __init__(self, input_dim=164, latent_dim=128, hidden_dim=256):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar

# Define the VAE loss function
def vae_loss_function(recon_x, x, mean, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_x, x.argmax(dim=-1))
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Model and training setup
input_dim = max_length * 21  # 20 amino acids + 1 padding, flattened sequence
latent_dim = 128
hidden_dim = 256
epochs = 42
learning_rate = 0.001

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

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Sampling from the latent space
vae.eval()
with torch.no_grad():
    for _ in range(5):  # Generate 5 sequences
        z = torch.randn(1, latent_dim)
        generated_sequence = vae.decode(z).squeeze().numpy()
        # Reshape from (max_length * 21) to (max_length, 21) and decode
        aa_sequence = decode_one_hot_sequence(generated_sequence.reshape(max_length, -1))
        # Remove padding tokens
        aa_sequence = aa_sequence.replace(PAD_TOKEN, "")
        print(aa_sequence)
