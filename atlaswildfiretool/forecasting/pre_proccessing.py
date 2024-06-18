import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "tag_sequences",
    "determine_threshold",
    "compute_mse",
    "detect_changes",
    "filter_train_images",
    "plot_differences_with_threshold",
    "analyze_sequences",
    "get_tags",
]


def determine_threshold(differences):
    """
    Determine the threshold for detecting significant changes.

    Parameters:
    differences (numpy array): An array of differences between consecutive images.

    Returns:
    float: The threshold value for detecting significant changes.
    """
    threshold = np.mean(differences) + np.std(differences)
    return threshold


def compute_mse(images):
    """
    Compute the Mean Squared Error (MSE) between consecutive images in a sequence.

    Parameters:
    images (np.ndarray): A 3D array of shape (num_images, height, width) containing the sequence of images.

    Returns:
    np.ndarray: A 1D array of MSE values, with each element representing the MSE between consecutive images.
    """
    mse_values = []
    for i in range(len(images) - 1):
        mse = np.mean((images[i] - images[i + 1]) ** 2)
        mse_values.append(mse)
    return np.array(mse_values)


def detect_changes(differences, threshold):
    """
    Detect change points where the differences exceed the threshold.

    Parameters:
    differences (numpy array): An array of differences between consecutive images.
    threshold (float): The threshold value for detecting significant changes.

    Returns:
    numpy array: An array of indices where significant changes are detected.
    """
    return np.where(differences > threshold)[0]


def tag_sequences(change_points, total_images):
    """
    Tag sequences based on detected change points.

    Parameters:
    change_points (numpy array): Indices of detected change points.
    total_images (int): Total number of images in the dataset.

    Returns:
    numpy array: An array of tags for each image.
    """
    # Initialize tags array with zeros
    tags = np.zeros(total_images, dtype=int)

    # Current tag
    current_tag = 1
    # Start of the current sequence
    start_index = 0

    # Assign tags
    for change_point in change_points:
        tags[start_index:change_point] = current_tag
        start_index = change_point
        current_tag += 1

    tags[start_index:] = current_tag

    return tags


def filter_train_images(train, percentile=99.2, save_path=None):
    """
    Filter train images based on the specified percentile.

    Parameters:
    train (np.ndarray): The original dataset of shape (num_images, height, width).
    percentile (float): Percentile to determine the threshold for filtering.
    save_path (str, optional): Path to save the filtered dataset as a .npy file. Default is None.

    Returns:
    np.ndarray: Filtered dataset.
    """
    # Compute differences (MSE values)
    differences = compute_mse(train)

    # Determine the threshold using the given percentile
    threshold = np.percentile(differences, percentile)

    # Detect change points
    change_points = np.where(differences > threshold)[0]

    # Generate tags for sequences
    tags = tag_sequences(change_points, train.shape[0])

    # Create a DataFrame from the tags
    df = pd.DataFrame({"index": np.arange(len(tags)), "wildfire number": tags})
    value_counts = df["wildfire number"].value_counts()
    useful_fire_indices = df[
        df["wildfire number"].isin(value_counts[value_counts == 100].index)
    ]["index"].values

    filtered_train = train[useful_fire_indices]

    # Save the filtered dataset if save_path is provided
    if save_path:
        np.save(save_path, filtered_train)
        print(f"Filtered dataset saved to {save_path}")

    return filtered_train


def plot_differences_with_threshold(test):
    """
    Plots the differences between consecutive images in the 'test' dataset and identifies change points.

    This function computes the sum of absolute differences between consecutive images in the provided dataset,
    determines an optimal threshold for detecting significant changes, and plots the differences along with
    the threshold. It also prints the optimal threshold, the indices of detected change points, and the number
    of change points detected.

    Parameters:
    test (list of numpy arrays): A list where each element is a numpy array representing an image. The images
                                 should be of the same shape.

    Returns:
    None

    Prints:
    Optimal threshold: The computed threshold for detecting change points.
    Detected change points at indices: The indices in the 'test' dataset where significant changes are detected.
    Number of change points detected: The total number of change points detected.

    Visualizes:
    A plot showing the differences between consecutive images, with the threshold indicated by a horizontal line.
    """

    differences = compute_mse(test)
    threshold = np.mean(differences) + np.std(differences)
    print("Optimal threshold:", threshold)

    change_points = np.where(differences > threshold)[0]

    print("Detected change points at indices:", change_points)
    print("Number of change points detected:", len(change_points))

    # Optional: Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(differences, label="Differences")
    plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
    plt.title("Differences Between Consecutive Images")
    plt.xlabel("Image Index")
    plt.ylabel("Sum of Absolute Differences")
    plt.legend()
    plt.show()


def analyze_sequences(tags):
    """
    Analyze sequences based on provided tags, including computing average sequence length
    and plotting the distribution of tags.

    Parameters:
    tags (numpy array): An array of tags for each data point.

    Returns:
    None

    Visualizes:
    A histogram showing the distribution of tags.
    """

    # Plot value count tags
    plt.figure(figsize=(8, 4))
    plt.hist(tags, bins=np.arange(tags.min(), tags.max() + 1) - 0.5, rwidth=0.5)
    plt.title("Distribution of Tags")
    plt.xlabel("Tag")
    plt.ylabel("Frequency")
    plt.show()


def get_tags(test):
    """
    Compute the tags for the given test dataset.

    Parameters:
    test (list of numpy arrays): A list where each element is a numpy array representing an image. The images
                                 should be of the same shape.

    Returns:
    numpy array: An array of tags for each image in the test dataset.
    """
    differences = compute_mse(test)
    threshold = determine_threshold(differences)
    print("Optimal threshold:", threshold)

    change_points = detect_changes(differences, threshold)

    tags = tag_sequences(change_points, len(test))
    return tags
