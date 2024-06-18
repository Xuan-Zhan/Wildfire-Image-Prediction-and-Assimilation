"""
This module contains functions and utilities for training and validating
generative Variational Autoencoder (VAE) models.

Functions:
    train: Trains the model for one epoch.
    validate: Validates the model.
    save_checkpoint: Saves the model checkpoint.
    train_vae: Trains the VAE model over multiple epochs, validates it,
                and saves checkpoints.
"""

from typing import List, Optional, Tuple

import torch
from livelossplot import PlotLosses
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    device: str,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    tepoch: tqdm,
    curr_epoch: int,
    n_epochs: int,
    logs: bool = True,
) -> Tuple[float, float, float]:
    """Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    device : str
        The device to run the model on, either 'cpu' or 'cuda'.
    data_loader : DataLoader
        The DataLoader for providing the training data.
    optimizer : optim.Optimizer
        The optimizer used for updating the weights.
    tepoch : tqdm
        The tqdm object for progress display.
    curr_epoch : int
        The current epoch number.
    n_epochs : int
        The total number of epochs to train.
    logs : bool, optional
        Flag to control the display of training progress, by default True.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the average training loss, mean squared error, and KL divergence for the epoch.
    """
    model.train()
    train_loss, mse_loss, kl_loss = 0.0, 0.0, 0.0
    for batch_idx, (X, _) in enumerate(data_loader, start=1):
        X = X.float().to(device)
        optimizer.zero_grad()
        X_recon, kl_div = model(X)
        mse = ((X - X_recon) ** 2).sum()
        loss = mse + kl_div
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        mse_loss += mse.item()
        kl_loss += kl_div.item()

        if logs:
            tepoch.set_description(
                f"Epoch: {curr_epoch}/{n_epochs} | Batch: {batch_idx}/{len(data_loader)}"
            )
            tepoch.set_postfix(loss=loss.item() / X.size(0), refresh=False)
            tepoch.update()

    return (
        train_loss / len(data_loader.dataset),
        mse_loss / len(data_loader.dataset),
        kl_loss / len(data_loader.dataset),
    )


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> Tuple[float, float, float]:
    """
    Validate the model.

    Parameters
    ----------
    model : nn.Module
        The neural network model to validate.
    data_loader : DataLoader
        The DataLoader for providing the validation data.
    device : str
        The device to run the model on, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the average validation loss, mean squared error, and KL divergence for the dataset.
    """
    model.eval()
    val_loss, mse_loss, kl_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.float().to(device)
            X_recon, kl_div = model(X)
            mse = ((X - X_recon) ** 2).sum()
            loss = mse + kl_div

            val_loss += loss.item()
            mse_loss += mse.item()
            kl_loss += kl_div.item()

    return (
        val_loss / len(data_loader.dataset),
        mse_loss / len(data_loader.dataset),
        kl_loss / len(data_loader.dataset),
    )


def save_checkpoint(
    model_weights: dict,
    optimizer_info: dict,
    model_save_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """
    Save the model checkpoint.

    Parameters
    ----------
    model_weights : dict
        The state dictionary of the model.
    optimizer_info : dict
        The state dictionary of the optimizer.
    model_save_path : str
        The file path to save the model checkpoint.
    epoch : int
        The current epoch number.
    train_loss : float
        The training loss at the current epoch.
    val_loss : float
        The validation loss at the current epoch.
    """
    torch.save(
        {
            "model_state_dict": model_weights,
            "optimizer_state_dict": optimizer_info,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        model_save_path,
    )


def train_vae(
    n_epochs: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model_save_path: Optional[str],
    use_liveloss: bool,
    device: str,
) -> Tuple[
    nn.Module,
    Tuple[List[float], List[float], List[float]],
    Tuple[List[float], List[float], List[float]],
]:
    """
    Train the VAE model over multiple epochs, validate it, and save checkpoints.

    Parameters
    ----------
    n_epochs : int
        The number of epochs to train the model.
    model : nn.Module
        The VAE model to train.
    optimizer : optim.Optimizer
        The optimizer for updating the model parameters.
    scheduler : optim.lr_scheduler._LRScheduler, optional
        The scheduler for adjusting the learning rate.
    train_loader : DataLoader
        The DataLoader for providing the training data.
    val_loader : Optional[DataLoader]
        The DataLoader for providing the validation data.
    model_save_path : Optional[str]
        The file path to save the model checkpoints.
    use_liveloss : bool
        Flag to control the use of LiveLossPlot for real-time loss plotting.
    device : str
        The device to run the model on, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[nn.Module, Tuple[List[float], List[float], List[float]], Tuple[List[float], List[float], List[float]]]
        A tuple containing the trained model, lists of training losses, and lists of validation losses.
    """

    liveloss = PlotLosses()
    train_losses, train_mse_losses, train_kldiv_losses = [], [], []
    val_losses, val_mse_losses, val_kldiv_losses = [], [], []

    with tqdm(
        total=len(train_loader) * n_epochs, desc="Training", unit="batch"
    ) as tepoch:
        for i in range(n_epochs):
            logs = {}
            _train_losses = train(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                tepoch=tepoch,
                curr_epoch=i,
                n_epochs=n_epochs,
                device=device,
                logs=True,
            )

            train_losses.append(_train_losses[0])
            train_mse_losses.append(_train_losses[1])
            train_kldiv_losses.append(_train_losses[2])
            logs["train loss"] = _train_losses[0]

            if val_loader:
                _val_losses = validate(
                    model=model, data_loader=val_loader, device=device
                )
                val_losses.append(_val_losses[0])
                val_mse_losses.append(_val_losses[1])
                val_kldiv_losses.append(_val_losses[2])
                logs["val loss"] = _val_losses[0]

            if model_save_path and (
                _train_losses[0] <= min(train_losses, default=float("inf"))
            ):
                save_checkpoint(
                    model_weights=model.state_dict(),
                    optimizer_info=optimizer.state_dict(),
                    model_save_path=model_save_path,
                    epoch=i,
                    train_loss=_train_losses[0],
                    val_loss=_val_losses[0],
                )

            if use_liveloss:
                liveloss.update(logs)
                liveloss.send()

            if scheduler:
                scheduler.step(_train_losses[0])

    return (
        model,
        (train_losses, train_mse_losses, train_kldiv_losses),
        (val_losses, val_mse_losses, val_kldiv_losses),
    )
