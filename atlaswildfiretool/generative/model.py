"""
This module contains classes and functions for training and evaluating
Variational Autoencoders (VAE) and Convolutional Variational Autoencoders
(CVAE).

Classes:
    VAE: Defines a basic Variational Autoencoder.
    GridSearchVAE: Performs grid search to find the best VAE model.
    Reshape: Utility module to reshape tensors.
    CVAE: Defines a Convolutional Variational Autoencoder.
"""

import itertools
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_image_dims (tuple): Dimensions of the input images
                                    (channels, height, width).
        latent_dims (int): Dimension of the latent space.
        hidden_layers (list): List of hidden layer sizes.
        activation (nn.Module, optional): Activation function to use.
                                            Defaults to nn.ReLU.
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """

    def __init__(
        self,
        input_image_dims: Tuple[int, int, int],
        latent_dims: int,
        hidden_layers: List[int],
        activation: Callable[..., nn.Module] = nn.ReLU,
        device: str = "cpu",
    ):
        super().__init__()

        # inputs.
        self.input_image_dims = input_image_dims
        self.c, self.h, self.w = input_image_dims
        self.hidden_layers = hidden_layers
        self.latent_dims = latent_dims
        self.device = device
        self.activation = activation
        self.distribution = torch.distributions.Normal(0, 1)

        # encoder layers.
        modules = []
        previous_dim = self.c * self.h * self.w
        for h_dim in hidden_layers:
            modules.append(nn.Linear(previous_dim, h_dim))
            modules.append(activation())
            previous_dim = h_dim
        self.encoder = nn.Sequential(*modules)

        self._mu = nn.Linear(hidden_layers[-1], self.latent_dims)
        self._logvar = nn.Linear(hidden_layers[-1], self.latent_dims)

        # decoder layers.
        modules = []
        current_dim = self.latent_dims
        for h_dim in reversed(hidden_layers):
            modules.append(nn.Linear(current_dim, h_dim))
            modules.append(activation())
            current_dim = h_dim

        modules.append(nn.Linear(hidden_layers[0], self.c * self.h * self.w))
        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input into the latent space using the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back into the input space.

        Args:
            x (torch.Tensor): Latent representation tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        return self.decoder(x)

    def sample_latent_space(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log variance of the latent space.

        Returns:
            tuple: Sampled latent vector and KL divergence.
        """
        sigma = torch.exp(0.5 * logvar)  # stability trick.
        z = mu + sigma * self.distribution.sample(mu.shape).to(self.device)
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z, kl_div

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Reconstructed tensor and KL divergence.
        """
        encoded = self.encode(x.view(x.size(0), -1))  # make sure its 1D.

        # get mu and logvar from latent space.
        mu = self._mu(encoded)
        logvar = self._logvar(encoded)

        # reparamaterise trick.
        z, kl_div = self.sample_latent_space(mu, logvar)

        decoded = self.decode(z).view(-1, self.c, self.h, self.w)

        return decoded, kl_div


class GridSearchVAE:
    """
    Grid search for finding the best VAE model parameters.

    Args:
        input_image_dims (tuple): Dimensions of the input images
                                (channels, height, width).
        hidden_layers (list): List of hidden layer sizes.
        latent_dims (list): List of latent dimensionalities to search.
        lrs (list): List of learning rates to search.
        batch_sizes (list): List of batch sizes to search.
        epochs (int, optional): Number of epochs to train each model.
                                Defaults to 10.
        device (str, optional): Device to run the models on. Defaults to 'cpu'.
    """

    def __init__(
        self,
        input_image_dims: Tuple[int, int, int],
        hidden_layers: List[int],
        latent_dims: List[int],
        lrs: List[float],
        batch_sizes: List[int],
        epochs: int = 10,
        device: str = "cpu",
    ):
        self.input_image_dims = input_image_dims
        self.latent_dims = latent_dims
        self.learning_rates = lrs
        self.batch_sizes = batch_sizes
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.device = device
        self.best_model = None
        self.best_params = None
        self.results = []
        self.num_combinations = None

    def fit(self, train_dataset, val_dataset) -> None:
        """
        Fits the VAE models with different parameter combinations.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset.
            val_dataset (torch.utils.data.Dataset): Validation dataset.
        """
        combinations = list(
            itertools.product(
                self.latent_dims,
                self.batch_sizes,
                self.hidden_layers,
                self.learning_rates,
            )
        )
        self.num_combinations = len(combinations)

        for latent_dim, batch_size, hidden_layers, lr in combinations:
            print(
                f"\n** Experiment: latent_dim={latent_dim},hidden_layers={hidden_layers}, batch_size={batch_size} @ lr={lr} **"
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # reinstantiate model with new combination of params.
            model = VAE(
                input_image_dims=self.input_image_dims,
                hidden_layers=hidden_layers,
                latent_dims=latent_dim,
                activation=nn.ReLU,  # fixing to ReLU() for grid search.
                device=self.device,
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

            train_losses, val_losses = [], []
            for i in range(self.epochs):
                with tqdm(
                    total=len(train_loader), desc="Training", unit="batch"
                ) as tepoch:
                    _train_losses = train(
                        model=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        tepoch=tepoch,
                        curr_epoch=i,
                        n_epochs=self.epochs,
                        device=self.device,
                        logs=True,
                    )

                    _val_losses = validate(
                        model=model,
                        data_loader=val_loader,
                        device=self.device,
                    )

                    train_losses.append(_train_losses[0])
                    val_losses.append(_val_losses[0])

            # store all results (and set best model if needed.).
            # naively taking the min of the training loss, would be worth
            # testing taking and average of the val and the train.
            # based on all the previous experiments, the FNN-VAE is not
            # overfitting and so the min MSE is an indication for the best
            # model given num epochs.
            min_training_loss = np.min(train_losses)
            min_validation_loss = np.min(val_losses)

            self.results.append(
                [
                    latent_dim,
                    batch_size,
                    hidden_layers,
                    lr,
                    min_training_loss,
                    min_validation_loss,
                ]
            )
            if (self.best_model is None) or (
                min_training_loss < self.best_params["train_loss"]
            ):
                self.best_params = {
                    "latent_dim": latent_dim,
                    "batch_size": batch_size,
                    "layers": hidden_layers,
                    "learning_rate": lr,
                    "train_loss": _train_losses[0],
                    "val_loss": _val_losses[0],
                    "epoch": i,
                }
                self.best_model = model

    def plot_results(self) -> None:
        """
        Plots the results of the grid search to identify best model parameters.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        latent_dims, batch_sizes, hidden_layers, lrs, train_losses, val_losses = zip(
            *self.results
        )
        ax.plot(range(self.num_combinations), train_losses, color="C0", marker="o")
        ax.plot(range(self.num_combinations), val_losses, color="C2", marker="o")
        ax.axhline(y=min(train_losses), color="C1", linestyle="--")
        ax.set_xticks(range(self.num_combinations))
        ax.set_xticklabels(
            [
                f"LD={ld}, BS: {bs}, LY: {lc}, LR: {lr}"
                for ld, bs, lc, lr in zip(latent_dims, batch_sizes, hidden_layers, lrs)
            ],
            rotation=90,
        )
        ax.set_xlabel("params")
        ax.set_ylabel("train loss - mse(reduction='sum')")
        ax.set_title("Grid Search Results")
        plt.show()


class Reshape(nn.Module):
    """
    Utility module to reshape tensors within a model.

    Args:
        *args: Desired shape of the tensor.
    """

    def __init__(self, *args: int):
        super().__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.shape)


class CVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (CVAE) class.

    Args:
        input_image_dims (tuple): Dimensions of the input images
                                    (channels, height, width).
        latent_dims (int): Dimensionality of the latent space.
        device (str): Device to run the model on.
        kernel_sizes (list): List of kernel sizes for convolutional layers.
        filter_sizes (list): List of filter sizes for convolutional layers.
        h_dim (int): Size of the hidden dimension.
        pool_size (int, optional): Size of the pooling layers. Defaults to 2.
    """

    def __init__(
        self,
        input_image_dims: Tuple[int, int, int],
        latent_dims: int,
        device: str,
        kernel_sizes: List[int],
        filter_sizes: List[int],
        h_dim: int,
        pool_size: int = 2,
    ):
        super().__init__()

        self.input_image_dims = input_image_dims
        self.c, self.h, self.w = input_image_dims
        self.latent_dims = latent_dims
        self.device = device

        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.h_dim = h_dim
        self.pool_size = pool_size

        self.activation = nn.Mish()
        self.distribution = torch.distributions.Normal(0, 1)

        self.encode = nn.Sequential(
            nn.Conv2d(
                in_channels=self.c,
                out_channels=self.filter_sizes[0],
                kernel_size=self.kernel_sizes[0],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[0]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_size),
            nn.Conv2d(
                in_channels=self.filter_sizes[0],
                out_channels=self.filter_sizes[1],
                kernel_size=self.kernel_sizes[1],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[1]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_size),
            nn.Conv2d(
                in_channels=self.filter_sizes[1],
                out_channels=self.filter_sizes[2],
                kernel_size=self.kernel_sizes[2],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[2]),
            self.activation,
            nn.Flatten(),
            nn.Linear(
                8 * 64 * 64, self.h_dim1
            ),  # TODO: workout flattened() size automatically.
            self.activation,
        )

        self._mu = nn.Linear(self.h_dim, self.latent_dims)
        self._logvar = nn.Linear(self.h_dim, self.latent_dims)

        self.decode = nn.Sequential(
            nn.Linear(self.latent_dims, self.h_dim),
            self.activation,
            nn.Linear(
                self.h_dim, 8 * 64 * 64
            ),  # TODO: workout flattened() size automatically.
            self.activation,
            Reshape(-1, 8, 64, 64),
            nn.Conv2d(
                in_channels=self.filter_sizes[2],
                out_channels=self.filter_sizes[1],
                kernel_size=self.kernel_sizes[2],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[1]),
            nn.Upsample(scale_factor=self.pool_size, mode="bilinear"),
            self.activation,
            nn.Conv2d(
                in_channels=self.filter_sizes[1],
                out_channels=self.filter_sizes[0],
                kernel_size=self.kernel_sizes[1],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[0]),
            nn.Upsample(scale_factor=self.pool_size, mode="bilinear"),
            self.activation,
            nn.Conv2d(
                in_channels=self.filter_sizes[0],
                out_channels=self.c,
                kernel_size=self.kernel_sizes[0],
                stride=1,
                padding="same",
            ),
            # nn.Upsample((256, 256), mode='bilinear'),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """
        Encodes the input into the latent space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        return self.encoder(x)

    def decode(self, x):
        """
        Decodes the latent representation back into the input space.

        Args:
            x (torch.Tensor): Latent representation tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        return self.decoder(x)

    def sample_latent_space(self, mu, logvar):
        """
        Samples from the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log variance of the latent space.

        Returns:
            tuple: Sampled latent vector and KL divergence.
        """
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * self.distribution.sample(mu.shape).to(self.device)
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z, kl_div

    def forward(self, x):
        """
        Forward pass through the CVAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Reconstructed tensor and KL divergence.
        """
        encoded = self.encode(x.flatten(start_dim=1))  # make sure its 1D.

        # get mu and logvar from latent space.
        mu = self._mu(encoded)
        logvar = self._logvar(encoded)

        # reparamaterise trick.
        z, kl_div = self.sample_latent_space(mu, logvar)

        decoded = self.decode(z)

        return decoded, kl_div
