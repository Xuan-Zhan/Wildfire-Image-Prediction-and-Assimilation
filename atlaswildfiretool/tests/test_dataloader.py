import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from atlaswildfiretool.forecasting.dataloader import CustomDataset


def test_custom_dataset_parameters():
    test_data_path = "atlaswildfiretool/tests/data/test_forecasting_dataset.npy"
    dataset = CustomDataset(data_file=test_data_path, sequence_length=4, step=1)

    assert str(test_data_path).endswith(".npy")
    assert dataset.data.shape[1:] == (3, 3)
    assert dataset.data_len > 0
    assert isinstance(dataset.data_len, int)


def test_custom_dataset_getitem_method():
    test_data_path = "atlaswildfiretool/tests/data/test_forecasting_dataset.npy"
    dataset = CustomDataset(data_file=test_data_path, sequence_length=4, step=1)
    actual_x, actual_y = dataset.__getitem__(0)

    desired_x = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
            [[3, 4, 5], [6, 7, 8], [9, 10, 11]],
            [[4, 5, 6], [7, 8, 9], [10, 11, 12]],
        ]
    )

    desired_y = np.array([[[5, 6, 7], [8, 9, 10], [11, 12, 13]]])

    assert isinstance(actual_x, torch.Tensor)
    assert isinstance(actual_y, torch.Tensor)
    assert np.allclose(np.array(actual_x)[:, 0, :, :], desired_x)  #
    assert np.allclose(np.array(actual_y), desired_y)
