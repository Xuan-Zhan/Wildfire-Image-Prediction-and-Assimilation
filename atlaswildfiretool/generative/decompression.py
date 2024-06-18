"""
This module contains classes for decompressing data for data assimilation
using different models, such as Variational Autoencoders (VAE) and Principal
Component Analysis (PCA).

Classes:
    Decompressor: An abstract class for decompressing data.
    VAEDecompressor: A derived class from Decompressor that handles a VAE's decompression.
    PCADecompressor: A derived class from Decompressor that handles a PCA's decompression.
"""

from abc import ABC, abstractmethod
from typing import Any, Union

import joblib
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


class Decompressor(ABC):
    """
    Abstract base class for decompressors.

    Methods:
        load_model(model_path): Load the model from the specified path.
        encode(x): Encode the input data.
        decode(x): Decode the encoded data.
    """

    def __init__(self):
        self.model: Any = None

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        pass

    @abstractmethod
    def encode(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        pass

    @abstractmethod
    def decode(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        pass


class VAEDecompressor(Decompressor):
    """
    Decompressor using a Variational Autoencoder (VAE).

    Args:
        model_path (str): Path to the VAE model file.
        model_obj (torch.nn.Module): VAE model object.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Methods:
        load_model(model_path, model_obj, device): Load the VAE model from the
        specified path.
        encode(x): Encode the input data using the VAE.
        decode(x): Decode the encoded data using the VAE.
    """

    def __init__(self, model_path: str, model_obj: Module, device: str):
        super().__init__()
        self.model_path: str = model_path
        self.model_obj: Module = model_obj
        self.device: str = device
        self.model = self.load_model(
            model_path=self.model_path, model_obj=self.model_obj, device=self.device
        )

    def load_model(self, model_path: str, model_obj: Module, device: str) -> Module:
        """
        Load the VAE model from the specified path.

        Args:
            model_path (str): Path to the VAE model file.
            model_obj (torch.nn.Module): VAE model object.
            device (str): Device to load the model on ('cpu' or 'cuda').

        Returns:
            torch.nn.Module: Loaded VAE model.
        """
        model_state = torch.load(model_path, map_location=torch.device(device))
        model_obj.load_state_dict(model_state["model_state_dict"])
        model_obj.to(device)
        model_obj.eval()
        return model_obj

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode the input data using the VAE.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Encoded data tensor.
        """
        return self.model.encode(x.to(self.device))

    def decode(self, x: Tensor) -> Tensor:
        """
        Decode the encoded data using the VAE.

        Args:
            x (torch.Tensor): Encoded data tensor.

        Returns:
            torch.Tensor: Decoded data tensor.
        """
        return self.model.decode(x.to(self.device))


class PCADecompressor(Decompressor):
    """
    Decompressor using Principal Component Analysis (PCA).

    Args:
        model_path (str): Path to the PCA model file.

    Methods:
        load_model(model_path): Load the PCA model from the specified path.
        encode(x): Encode the input data using the PCA model.
        decode(x): Decode the encoded data using the PCA model.
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path: str = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load the PCA model from the specified path.

        Args:
            model_path (str): Path to the PCA model file.

        Returns:
            sklearn.decomposition.PCA: Loaded PCA model.
        """
        return joblib.load(model_path)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode the input data using the PCA model.

        Args:
            x (numpy.ndarray): Input data array.

        Returns:
            numpy.ndarray: Encoded data array.
        """
        return self.model.transform(x)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Decode the encoded data using the PCA model.

        Args:
            x (numpy.ndarray): Encoded data array.

        Returns:
            numpy.ndarray: Decoded data array.
        """
        return self.model.inverse_transform(x)
