import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class CNNAutoencoder(nn.Module):
    """
    This module implements a convolutional neural network (CNN) based autoencoder designed for data compression and
    decompression tasks. It provides functionalities for training the autoencoder on specific datasets, saving and loading
    models, and evaluating the model's performance through mean squared error (MSE) calculations and visualizations of
    compression results.

    The CNNAutoencoder class encapsulates the entire workflow needed for processing data using CNN architectures within the
    autoencoder paradigm. This includes methods for loading data, training the model, and assessing the quality of the
    compression. The class is equipped to handle both training and test datasets, along with specific 'background' and
    'observation' data sets used in evaluations.

    Features:
        - Efficient data handling with methods to load various data formats.
        - A structured training method that logs losses and supports visual evaluation of the model's learning progress.
        - Model persistence capabilities, allowing the trained model to be saved and reloaded.
        - Evaluation tools that measure the model's performance in both its encoded (compressed) and decoded (reconstructed)
          states, including MSE calculations and the option to visualize the original and reconstructed images.

    Example usage:
        autoencoder = CNNAutoencoder(train_path='path_to_train_data.npy', test_path='path_to_test_data.npy',
                                     back_path='path_to_background_data.npy', obs_path='path_to_observation_data.npy',
                                     device='cuda')
        autoencoder.train_model(num_epochs=50, lr=0.001)
        autoencoder.save_model('path_to_save_trained_model.pth')
        autoencoder.load_model('path_to_load_trained_model.pth')
        mse_values = autoencoder.compute_mse()
        autoencoder.compress_and_decompress(visualize=True)

    This class is intended for researchers and developers working in the fields of data compression, image processing, or
    machine learning, providing a flexible tool for experimenting with CNN-based autoencoders on custom datasets.
    """

    def __init__(self, train_path, test_path, back_path, obs_path, device="cpu"):
        super(CNNAutoencoder, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.back_path = back_path
        self.obs_path = obs_path
        self.device = device

        # load data
        self.train_tensor, self.test_tensor = self.load_train_test_data()
        self.back_tensor, self.obs_tensor = self.load_back_obs_data()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_train_test_data(self):
        train_data = np.load(self.train_path)
        test_data = np.load(self.test_path)
        train_tensor = (
            torch.tensor(train_data).float().unsqueeze(1)
            if train_data.ndim == 3
            else torch.tensor(train_data).float()
        )
        test_tensor = (
            torch.tensor(test_data).float().unsqueeze(1)
            if test_data.ndim == 3
            else torch.tensor(test_data).float()
        )
        return train_tensor, test_tensor

    def load_back_obs_data(self):
        back_data = np.load(self.back_path)
        obs_data = np.load(self.obs_path)
        if back_data.ndim == 2:  # if shape (256, 256)
            back_data = np.expand_dims(back_data, axis=0)  # convert to (1, 256, 256)
        if obs_data.ndim == 2:  # if shape (256, 256)
            obs_data = np.expand_dims(obs_data, axis=0)  # convert to (1, 256, 256)
        back_tensor = (
            torch.tensor(back_data).float().unsqueeze(1)
            if back_data.ndim == 3
            else torch.tensor(back_data).float()
        )
        obs_tensor = (
            torch.tensor(obs_data).float().unsqueeze(1)
            if obs_data.ndim == 3
            else torch.tensor(obs_data).float()
        )
        return back_tensor, obs_tensor

    def train_model(self, num_epochs=200, lr=1e-3, batch_size=256):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        train_dataset = TensorDataset(self.train_tensor)
        test_dataset = TensorDataset(self.test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            for data in train_loader:
                inputs = data[0].to(self.device)
                optimizer.zero_grad()
                encoded, decoded = self(inputs)
                loss = criterion(decoded, inputs)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_losses.append(epoch_train_loss / len(train_loader))

            self.eval()
            epoch_test_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    inputs = data[0].to(self.device)
                    encoded, decoded = self(inputs)
                    loss = criterion(decoded, inputs)
                    epoch_test_loss += loss.item()

            test_losses.append(epoch_test_loss / len(test_loader))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}"
                )

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Testing Loss")
        plt.show()

    def compute_mse(self, n=None):
        criterion = torch.nn.MSELoss()
        self.eval()
        with torch.no_grad():
            back_encoded, back_decoded = self(self.back_tensor.to(self.device))
            obs_encoded, obs_decoded = self(self.obs_tensor.to(self.device))
            if n is not None:
                back_encoded, back_decoded = back_encoded[:n], back_decoded[:n]
                obs_encoded, obs_decoded = obs_encoded[:n], obs_decoded[:n]
            mse_reduced_space = criterion(back_encoded, obs_encoded).item()
            mse_physical_space = criterion(
                back_decoded,
                (
                    self.obs_tensor[:n].to(self.device)
                    if n is not None
                    else self.obs_tensor.to(self.device)
                ),
            ).item()
        print(f"MSE in reduced space: {mse_reduced_space:.4f}")
        print(f"MSE in physical space: {mse_physical_space:.4f}")
        return mse_reduced_space, mse_physical_space

    def compress_and_decompress(self, visualize=False, n=None):
        self.eval()
        with torch.no_grad():
            back_encoded, back_decoded = self(self.back_tensor.to(self.device))
            obs_encoded, obs_decoded = self(self.obs_tensor.to(self.device))

        if n is None:
            n = min(self.back_tensor.shape[0], self.obs_tensor.shape[0])
        n = min(n, self.back_tensor.shape[0], self.obs_tensor.shape[0])

        if visualize:
            plt.figure(figsize=(10, 4))
            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(self.back_tensor[i].cpu().numpy().squeeze(), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Original")

                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(back_decoded[i].cpu().numpy().squeeze(), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Reconstructed")

            plt.show()

            plt.figure(figsize=(10, 4))
            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(self.obs_tensor[i].cpu().numpy().squeeze(), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Original")

                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(obs_decoded[i].cpu().numpy().squeeze(), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Reconstructed")

            plt.show()

        return back_encoded[:n], back_decoded[:n], obs_encoded[:n], obs_decoded[:n]
