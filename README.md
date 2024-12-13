# ProtVAE

This project implements a Variational Autoencoder (VAE) to encode and generate protein sequences using a one-hot encoded representation of amino acids.

## Features
- Supports multi-sequence FASTA files for input.
- Uses a one-hot encoding scheme for amino acids, including padding.
- Trains a VAE to learn a latent representation of protein sequences.
- Generates new protein-like sequences by sampling from the latent space.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/annadiarov/ProtVAE
   cd ProtVAE
   ```
2. Install the required dependencies (we recommend using a virtual environment
   and install pytorch using the instructions from the [official website](https://pytorch.org/)):
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage
### Training
To train the VAE, run the following command:
```bash
python src/training.py --fasta_file data/example.fasta --epochs 50 --batch_size 32
```
This will train the VAE on the sequences in `data/example.fasta` for 50 epochs 
using a batch size of 32.
The trained weights will be saved as `vae_weights.pth` by default but can be 
changed using the `--output-weights` argument.

### Generating Sequences
After training the VAE, you can generate new sequences by running:
```bash
python src/sampling.py --weights vae_weights.pth --num_samples 10
```
This will generate 10 new sequences using the trained VAE weights in 
`generated_sequences.fasta` by default, but this can be changed using the 
`--output-file` argument.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
