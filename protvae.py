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


def vae_loss_function(recon_x, x, mean, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_x, x.argmax(dim=-1))
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_div


input_dim = 164  # 20 amino acids + padding, length of sequences
latent_dim = 128
hidden_dim = 256
batch_size = 32
epochs = 42
learning_rate = 0.001



vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

for epoch in range(epochs):
    vae.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        recon_x, mean, logvar = vae(batch)
        loss = vae_loss_function(recon_x, batch, mean, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


vae.eval()
with torch.no_grad():
    for _ in range(n):  
        z = torch.randn(1, latent_dim)
        generated_sequence = vae.decode(z).squeeze().numpy()


