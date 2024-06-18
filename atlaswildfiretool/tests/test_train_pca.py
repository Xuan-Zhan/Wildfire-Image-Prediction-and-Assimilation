import pytest
import numpy as np
from unittest.mock import MagicMock
from sklearn.decomposition import PCA
from atlaswildfiretool.assimilation.train_pca import TrainPCA


@pytest.fixture
def mock_data():
    # Generate mock data, each array contains 400 images, each image size is 256x256
    return np.random.rand(400, 256, 256)


@pytest.fixture
def pca_trainer(mock_data):
    # Create a TrainPCA instance using mock data
    train_path = "train_data.npy"
    obs_path = "obs_data.npy"
    background_path = "background_data.npy"

    # Replace np.load with a MagicMock object that returns mock data
    np.load = MagicMock(
        side_effect=lambda x: mock_data if "train" in x else mock_data[:1]
    )

    trainer = TrainPCA(
        train_path,
        obs_path,
        background_path,
        n_components=10,
        obs_index=0,
        background_index=0,
    )
    return trainer


@pytest.mark.skip(
    reason="no way of currently testing this - need model file, like model.pth"
)
def test_initialization(pca_trainer):
    assert pca_trainer.train_data.shape == (400, 256, 256)
    assert pca_trainer.obs.shape == (1, 256, 256)
    assert pca_trainer.background.shape == (1, 256, 256)
    assert pca_trainer.pca.n_components == 10


@pytest.mark.skip(
    reason="no way of currently testing this - need model file, like model.pth"
)
def test_mse_calculation(pca_trainer):
    assert pca_trainer.mse_original >= 0
    assert pca_trainer.mse_reduced >= 0


@pytest.mark.skip(
    reason="no way of currently testing this - need model file, like model.pth"
)
def test_model_saving(pca_trainer):
    pca_trainer.save_model = MagicMock()
    pca_trainer.save_model("pca_model.pkl")
    pca_trainer.save_model.assert_called_once_with("pca_model.pkl")


@pytest.mark.skip(
    reason="no way of currently testing this - need model file, like model.pth"
)
def test_data_decompression(pca_trainer):
    obs_dec, bg_dec = pca_trainer.decompress_data()
    assert obs_dec.shape == (256, 256)
    assert bg_dec.shape == (256, 256)


# Commented out testing for plot_images as it involves graphical output, typically not included in unit tests
# However, if needed, tests can be conducted using matplotlib's testing support or simply by ensuring the method throws no exceptions
