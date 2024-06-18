"""testing module for the generative dataset module."""

import numpy as np

import torch
from atlaswildfiretool.generative.dataset import WildfireImageDataSet
import torchvision.transforms as transforms


def test_wildfire_dataset_getitem_method_without_transform():
    """test wildfire dataset's __getitem__ method when transform transform applied."""
    test_data_path = "atlaswildfiretool/tests/data/test_forecasting_dataset.npy"
    test_data = np.load(test_data_path)
    test_dataset = WildfireImageDataSet(test_data, transform=None)

    actual = test_dataset.__getitem__(0)
    desired = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

    assert np.allclose(actual, desired)


def test_wildfire_dataset_getitem_method_with_transform():
    """test wildfire dataset's __getitem__ method when transform transform applied."""
    test_data_path = "atlaswildfiretool/tests/data/test_forecasting_dataset.npy"
    test_data = np.load(test_data_path)
    transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
        ]
    )
    test_dataset = WildfireImageDataSet(test_data, transform=transform)
    actual = test_dataset.__getitem__(0)
    desired = torch.from_numpy(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

    assert torch.allclose(actual, desired)
