import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(
        self, hidden_dim, train_path, test_path, back_path, obs_path, device="cpu"
    ):
        super(Autoencoder, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.back_path = back_path
        self.obs_path = obs_path
        self.device = device

        # 加载数据
        self.train_tensor, self.test_tensor = self.load_train_test_data()
        self.back_tensor, self.obs_tensor = self.load_back_obs_data()

        self.input_dim = self.train_tensor.shape[1]  # 已经展平，不需要再乘

        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, self.input_dim), nn.ReLU(True)
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
        self.eval()  # Set the model to evaluation mode after loading

    def load_train_test_data(self):
        train_data = np.load(self.train_path)
        test_data = np.load(self.test_path)
        train_tensor = torch.tensor(train_data).float().view(train_data.shape[0], -1)
        test_tensor = torch.tensor(test_data).float().view(test_data.shape[0], -1)
        return train_tensor, test_tensor

    def load_back_obs_data(self):
        back_data = np.load(self.back_path)
        obs_data = np.load(self.obs_path)
        if back_data.ndim == 2:  # 如果形状是 (256, 256)
            back_data = np.expand_dims(back_data, axis=0)  # 变成 (1, 256, 256)
        if obs_data.ndim == 2:  # 如果形状是 (256, 256)
            obs_data = np.expand_dims(obs_data, axis=0)  # 变成 (1, 256, 256)
        back_tensor = torch.tensor(back_data).float().view(back_data.shape[0], -1)
        obs_tensor = torch.tensor(obs_data).float().view(obs_data.shape[0], -1)
        return back_tensor, obs_tensor

    def train_model(self, num_epochs=200, lr=1e-3):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            encoded, decoded = self(self.train_tensor.to(self.device))
            loss = criterion(decoded, self.train_tensor.to(self.device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            self.eval()
            with torch.no_grad():
                _, test_decoded = self(self.test_tensor.to(self.device))
                test_loss = criterion(
                    test_decoded, self.test_tensor.to(self.device)
                ).item()
                test_losses.append(test_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}"
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
        self.eval()  # Ensure the model is in evaluation mode
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
                plt.imshow(
                    self.back_tensor[i].cpu().numpy().reshape(256, 256), cmap="gray"
                )
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Original")

                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(back_decoded[i].cpu().numpy().reshape(256, 256), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Reconstructed")

            plt.show()

            plt.figure(figsize=(10, 4))
            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(
                    self.obs_tensor[i].cpu().numpy().reshape(256, 256), cmap="gray"
                )
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Original")

                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(obs_decoded[i].cpu().numpy().reshape(256, 256), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title("Reconstructed")

            plt.show()

        return back_encoded[:n], back_decoded[:n], obs_encoded[:n], obs_decoded[:n]
