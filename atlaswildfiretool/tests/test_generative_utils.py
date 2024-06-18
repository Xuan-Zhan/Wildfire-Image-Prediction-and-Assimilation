"""testing module for the generative utils.py."""

import numpy as np

from atlaswildfiretool.generative.utils import (
    sequential_train_val_split,
    sequential_undersample_3d_arr,
)


def test_test_sequential_train_val_split():
    """test sequential_train_val_split() for ~ 90:10 split."""
    test_data_path = "atlaswildfiretool/tests/data/test_forecasting_dataset.npy"
    test_data = np.load(test_data_path)

    actual_train, actual_val = sequential_train_val_split(
        train_data=test_data, sequence_jump=3, start_offset=2, jump_multiplier=3
    )

    desired_val_indexes = [i for i in range(7, 50, 9)]
    desired_train_indexes = [
        i for i in range(test_data.shape[0]) if i not in desired_val_indexes
    ]
    desired_train = test_data[desired_train_indexes]
    desired_val = test_data[desired_val_indexes]

    assert np.allclose(actual_train, desired_train)
    assert np.allclose(actual_val, desired_val)


def test_sequential_undersample_3d_arr_no_jump():
    """Test sequential_undersample_3d_arr() when jump is < len arr."""
    test_arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])
    actual = sequential_undersample_3d_arr(arr=test_arr, sequence_jump=1, _print=False)
    desired = np.array(
        [[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]]
    )  # same as original.

    assert np.allclose(actual, desired)


def test_sequential_undersample_3d_arr_jump_size_within_len_arr():
    """Test sequential_undersample_3d_arr() when jump is < len arr."""
    test_arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])
    actual = sequential_undersample_3d_arr(arr=test_arr, sequence_jump=2, _print=False)
    desired = np.array([[[1, 2, 3]], [[4, 5, 6]]])

    assert np.allclose(actual, desired)


def test_sequential_undersample_3d_arr_jump_size_outside_len_arr():
    """Test sequential_undersample_3d_arr() when jump is > len arr."""
    test_arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])
    actual = sequential_undersample_3d_arr(
        arr=test_arr, sequence_jump=100, _print=False
    )
    desired = np.array([[[1, 2, 3]]])  # only 1st row by default.

    assert np.allclose(actual, desired)
