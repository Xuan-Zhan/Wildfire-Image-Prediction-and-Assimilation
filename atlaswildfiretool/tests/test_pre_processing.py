import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from atlaswildfiretool.forecasting.pre_proccessing import (
    analyze_sequences,
    compute_mse,
    detect_changes,
    determine_threshold,
    filter_train_images,
    get_tags,
    plot_differences_with_threshold,
    tag_sequences,
)


@pytest.fixture
def test_data():
    test_data_path = "atlaswildfiretool/tests/data/test_forecasting_dataset.npy"
    return np.load(test_data_path)


def test_determine_threshold(test_data):
    differences = compute_mse(test_data)
    threshold = determine_threshold(differences)

    assert isinstance(threshold, float)
    assert threshold > 0


def test_compute_mse(test_data):
    mse_values = compute_mse(test_data)

    assert isinstance(mse_values, np.ndarray)
    assert mse_values.shape == (test_data.shape[0] - 1,)
    assert np.all(mse_values >= 0)


def test_detect_changes(test_data):
    differences = compute_mse(test_data)
    threshold = determine_threshold(differences)
    change_points = detect_changes(differences, threshold)

    assert isinstance(change_points, np.ndarray)
    assert np.all(change_points >= 0)
    assert np.all(change_points < len(differences))


def test_tag_sequences(test_data):
    differences = compute_mse(test_data)
    threshold = determine_threshold(differences)
    change_points = detect_changes(differences, threshold)
    tags = tag_sequences(change_points, len(test_data))

    assert isinstance(tags, np.ndarray)
    assert len(tags) == len(test_data)
    assert np.all(tags >= 0)


def test_filter_train_images(test_data):
    filtered_train = filter_train_images(test_data, percentile=90)

    assert isinstance(filtered_train, np.ndarray)
    assert filtered_train.shape[1:] == (3, 3)


# def test_plot_differences_with_threshold(test_data, capsys):
#     plot_differences_with_threshold(test_data)
#     captured = capsys.readouterr()

#     assert "Optimal threshold:" in captured.out
#     assert "Detected change points at indices:" in captured.out

# def test_analyze_sequences(test_data):
#     tags = get_tags(test_data)
#     analyze_sequences(tags)

#     assert len(tags) == len(test_data)


def test_get_tags(test_data):
    tags = get_tags(test_data)

    assert isinstance(tags, np.ndarray)
    assert len(tags) == len(test_data)
