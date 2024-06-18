from abc import abstractmethod
import joblib
import torch
import numpy as np

"""
This Python module offers a framework for decompression models leveraging techniques such as PCA and autoencoders.
It includes abstract and concrete classes to handle the loading, encoding, and decoding of data, catering to different
model types and data transformation requirements.

Classes:
    - Decompressor: An abstract base class that defines the methods load_model, encode, and decode. All decompressor
      implementations must inherit from this class and provide these method implementations.
    - PCADecompressor: Implements PCA-based decompression. It is optimized for simple linear transformations and
      dimensionality reduction.
    - AutoencoderDecompressor: Implements decompression using a neural network-based autoencoder, ideal for complex,
      nonlinear transformations. It uses PyTorch and supports computation on GPUs.
    - CAEDecompressor: Another variant of AutoencoderDecompressor that can handle different image and tensor sizes, also
      utilizing PyTorch for operations.

Usage:
    To use a decompressor, instantiate it with the required model path, and optionally, the model object and device
    configuration. Then, use the encode and decode methods to transform and revert data, respectively.

Example:
    # Example of PCA decompression
    pca_decompressor = PCADecompressor(model_path='path_to_pca_model.pkl')
    encoded_data = pca_decompressor.encode(np.random.rand(1, 256, 256))
    decoded_data = pca_decompressor.decode(encoded_data)

    # Example of using an Autoencoder for decompression
    from my_autoencoder_model import AutoencoderModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autoencoder_model = AutoencoderModel()
    autoencoder_decompressor = AutoencoderDecompressor(model_path='path_to_autoencoder_model.pth', model_obj=autoencoder_model, device=device)
    encoded_data = autoencoder_decompressor.encode(np.random.rand(1, 256, 256))
    decoded_data = autoencoder_decompressor.decode(encoded_data)

Notes:
    - Ensure that PCA models are used on systems without the need for GPU computation.
    - Autoencoder-based decompressors should be configured with the correct device (CPU or GPU) to optimize performance.
    - Validate all model paths and ensure correct setup before utilization to avoid runtime errors.
"""


class Decompressor:
    def __init__(self):
        self.model = None

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass


class PCADecompressor(Decompressor):
    # non CPU/GPU --> dont need device.
    def __init__(self, model_path):
        super().__init__()

        self.model_path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model_obj = joblib.load(model_path)
        return model_obj

    def encode(self, x):
        if x.ndim == 2:  # has only two dimensions, i.e. the shape is (256, 256)
            x = np.expand_dims(x, axis=0)  # Add a dimension to become (1, 256, 256)

        x = x.reshape(x.shape[0], -1)  # Convert to (n, 256*256)

        return self.model.transform(x)

    def decode(self, x):
        return self.model.inverse_transform(x).reshape(1, 256, 256)

    def get_model_type(self):
        return "PCA"


class AutoencoderDecompressor(Decompressor):
    def __init__(self, model_path, model_obj, device):
        super().__init__()
        self.model_path = model_path
        self.model_obj = model_obj
        self.device = device
        self.model = self.load_model(
            model_path=model_path, model_obj=self.model_obj, device=self.device
        )

    def load_model(self, model_path, model_obj, device):
        try:
            model_state = torch.load(model_path, map_location=torch.device(device))
            model_obj.load_state_dict(model_state)
            model_obj.to(device)
            model_obj.eval()
            return model_obj
        except KeyError:
            raise ValueError("Model state dict key not found in the loaded model file")

    def encode(self, x):
        if x.ndim == 2:  # If the input shape is (256, 256)
            x = np.expand_dims(x, axis=0)  # covert to (1, 256, 256)
        x_tensor = torch.tensor(x).float().view(x.shape[0], -1)
        encoded = self.model.encoder(x_tensor.to(self.device))
        return encoded.cpu().detach().numpy()  # convert to numpy

    def decode(self, x):
        x_tensor = torch.tensor(x).float().to(self.device)  # covert numpy to tensor
        decoded = self.model.decoder(x_tensor)
        decoded = decoded.view(decoded.size(0), 256, 256)  # flat back
        return decoded.cpu().detach().numpy()  # convert to numpy

    def get_model_type(self):
        return "AutoEncoder"


class CAEDecompressor(Decompressor):
    def __init__(self, model_path, model_obj, device):
        super().__init__()
        self.model_path = model_path
        self.model_obj = model_obj
        self.device = device
        self.model = self.load_model(
            model_path=model_path, model_obj=self.model_obj, device=self.device
        )

    def load_model(self, model_path, model_obj, device):
        try:
            model_state = torch.load(model_path, map_location=torch.device(device))
            model_obj.load_state_dict(model_state)
            model_obj.to(device)
            model_obj.eval()
            return model_obj
        except KeyError:
            raise ValueError("Model state dict key not found in the loaded model file")

    def encode(self, x):
        if x.ndim == 2:  # if shape (256, 256)
            x = np.expand_dims(x, axis=0)  # convert to (1, 256, 256)
        if x.ndim == 3:  # if shape (N, 256, 256)
            x = np.expand_dims(x, axis=1)  # convert to (N, 1, 256, 256)
        x_tensor = torch.tensor(x).float().to(self.device)
        encoded = self.model.encoder(x_tensor)
        encoded = (
            encoded.cpu().detach().numpy()
        )  # Correct conversion from tensor to numpy array

        # Assuming the output of encoder is (N, C, H, W)
        if len(encoded.shape) == 4:
            new_shape = (
                encoded.shape[0],
                encoded.shape[1] * encoded.shape[2] * encoded.shape[3],
            )
            encoded = encoded.reshape(new_shape)
        return encoded

    def decode(self, x):
        # check input shape，convert to (N, C, H, W) 如果需要
        if len(x.shape) == 2:  # if input is (N, C*H*W)
            C = 8  # chanel
            H = 64
            W = 64
            x = x.reshape(-1, C, H, W)
        x_tensor = torch.tensor(x).float().to(self.device)  # convert numpy to tensor
        decoded = self.model.decoder(x_tensor)
        decoded = decoded.squeeze(1)  # delet chanel
        return decoded.cpu().detach().numpy()  # convert to numpy

    def get_model_type(self):
        return "CAutoEncoder"
