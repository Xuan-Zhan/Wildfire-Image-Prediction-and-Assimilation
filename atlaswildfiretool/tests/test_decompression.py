import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from atlaswildfiretool.assimilation.decompression import (
    PCADecompressor,
    AutoencoderDecompressor,
    CAEDecompressor,
)


# Mock data for testing
@pytest.fixture
def mock_data():
    return np.random.rand(1, 256, 256)  # Assuming the input data is a 256x256 image


# Setup PCA decompressor with mock
@pytest.fixture
def pca_decompressor():
    model_path = "pca_model_task2.pkl"  # The model path should point to the correct model file in actual use
    decompressor = PCADecompressor(model_path)
    decompressor.model = MagicMock()
    decompressor.model.transform = MagicMock(
        return_value=np.random.rand(1, 80)
    )  # Assuming n_components=80 for PCA
    decompressor.model.inverse_transform = MagicMock(
        return_value=np.random.rand(1, 256 * 256).flatten()
    )
    return decompressor


# Setup Autoencoder decompressor with mock
@pytest.fixture
def autoencoder_decompressor():
    model_path = "autoencoder_class_model.pth"  # The model path should point to the correct model file in actual use
    model_obj = MagicMock()
    decompressor = AutoencoderDecompressor(model_path, model_obj, "cpu")
    decompressor.model = model_obj
    model_obj.encoder = MagicMock(
        return_value=torch.rand(1, 512)
    )  # Assuming encoder output is 512 features
    model_obj.decoder = MagicMock(return_value=torch.rand(1, 1, 256, 256))
    return decompressor


# Setup CAE decompressor with mock
@pytest.fixture
def cae_decompressor():
    model_path = "cnn_model.pth"  # The model path should point to the correct model file in actual use
    model_obj = MagicMock()
    decompressor = CAEDecompressor(model_path, model_obj, "cpu")
    decompressor.model = model_obj
    model_obj.encoder = MagicMock(
        return_value=torch.rand(1, 8, 64, 64)
    )  # Assuming the encoder's output shape is (N, C, H, W)
    model_obj.decoder = MagicMock(return_value=torch.rand(1, 1, 256, 256))
    return decompressor


# Test PCA Decompressor
@pytest.mark.skip(
    reason="no way of currently testing this - need model file, like model.pth"
)
def test_pca_decompressor_encoding_decoding(pca_decompressor, mock_data):
    encoded_data = pca_decompressor.encode(mock_data)
    decoded_data = pca_decompressor.decode(encoded_data)
    assert encoded_data.shape == (1, 80), "Encoded data should have shape (1, 80)"
    assert decoded_data.shape == (
        1,
        256,
        256,
    ), "Decoded data should have shape (1, 256, 256)"


# Test Autoencoder Decompressor
@pytest.mark.skip(
    reason="no way of currently testing this - need model file, like model.pth"
)
def test_autoencoder_decompressor_encoding_decoding(
    autoencoder_decompressor, mock_data
):
    encoded_data = autoencoder_decompressor.encode(mock_data)
    decoded_data = autoencoder_decompressor.decode(encoded_data)
    assert encoded_data.shape == (1, 512), "Encoded data should have shape (1, 512)"
    assert decoded_data.shape == (
        1,
        256,
        256,
    ), "Decoded data should have original shape (1, 256, 256)"


# Test CAE Decompressor
@pytest.mark.skip(
    reason="no way of currently testing this - need model file, like model.pth"
)
def test_cae_decompressor_encoding_decoding(cae_decompressor, mock_data):
    encoded_data = cae_decompressor.encode(mock_data)
    decoded_data = cae_decompressor.decode(encoded_data)
    assert encoded_data.shape == (
        1,
        8 * 64 * 64,
    ), "Encoded data should be flattened to shape (1, 8*64*64)"
    assert decoded_data.shape == (
        1,
        256,
        256,
    ), "Decoded data should have original shape (1, 256, 256)"
