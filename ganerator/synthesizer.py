import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class GANDataSynthesizer:
    def __init__(
        self, input_dim, output_dim, latent_dim, hidden_dim, num_epochs=100, batch_size=64, lr=0.001
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.discriminator = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.criterion = nn.BCELoss()

    def build_generator(self):
        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)

    def build_discriminator(self):
        self.discriminator = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def preprocess_data(self, data):
        # Ensure CNT_CHILDREN is positive integer
        data[:, 0] = np.clip(data[:, 0], 0, None)
        data[:, 0] = np.round(data[:, 0]).astype(int)

        # Ensure AMT_INCOME_TOTAL and AMT_CREDIT are greater than or equal to zero
        data[:, 1] = np.clip(data[:, 1], 0, None)
        data[:, 2] = np.clip(data[:, 2], 0, None)

        return data

    def train(self, real_data):
        real_data = self.preprocess_data(real_data)
        real_data = torch.tensor(real_data, dtype=torch.float32).to(self.device)

        # Initialize models
        self.build_generator()
        self.build_discriminator()

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.num_epochs):
            for i in range(0, real_data.size(0), self.batch_size):
                # Train discriminator
                self.optimizer_D.zero_grad()
                # Real data
                real_batch = real_data[i : i + self.batch_size]
                real_labels = torch.ones(real_batch.size(0), 1).to(self.device)
                real_output = self.discriminator(real_batch)
                d_loss_real = self.criterion(real_output, real_labels)
                d_loss_real.backward()
                # Fake data
                noise = torch.randn(real_batch.size(0), self.latent_dim).to(self.device)
                fake_batch = self.generator(noise)
                fake_labels = torch.zeros(fake_batch.size(0), 1).to(self.device)
                fake_output = self.discriminator(fake_batch.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                d_loss_fake.backward()
                self.optimizer_D.step()
                # Train generator
                self.optimizer_G.zero_grad()
                noise = torch.randn(real_batch.size(0), self.latent_dim).to(self.device)
                gen_batch = self.generator(noise)
                gen_output = self.discriminator(gen_batch)
                g_loss = self.criterion(gen_output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

    def generate_samples(self, num_samples):
        noise = torch.randn(num_samples, self.latent_dim).to(self.device)
        generated_samples = self.generator(noise).detach().cpu().numpy()
        return self.preprocess_data(generated_samples)
