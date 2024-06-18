import matplotlib.pyplot as plt
import numpy as np
import cv2


def augment_image(image):
    augmented_images = {
        "rotated_90": [],
        "rotated_180": [],
        "rotated_270": [],
        "flipped_h": [],
        "flipped_v": [],
        "translated": [],
        "scaled": [],
    }

    # 旋转
    rows, cols = image.shape
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images[f"rotated_{angle}"].append(rotated)

    # 翻转
    augmented_images["flipped_h"].append(cv2.flip(image, 1))  # 水平翻转
    augmented_images["flipped_v"].append(cv2.flip(image, 0))  # 垂直翻转

    # 平移
    for tx, ty in [(10, 0), (-10, 0), (0, 10), (0, -10)]:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images["translated"].append(translated)

    # 缩放
    for scale in [0.9, 1.1]:
        resized = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        # 确保调整后的图像大小一致
        if resized.shape != image.shape:
            resized = cv2.resize(resized, (cols, rows), interpolation=cv2.INTER_LINEAR)
        augmented_images["scaled"].append(resized)

    return augmented_images


def augment_images_in_sequence(images):
    all_augmented_images = {
        "rotated_90": [],
        "rotated_180": [],
        "rotated_270": [],
        "flipped_h": [],
        "flipped_v": [],
        "translated": [],
        "scaled": [],
    }

    for image in images:
        augmented = augment_image(image)
        for key in all_augmented_images.keys():
            all_augmented_images[key].extend(augmented[key])

    return all_augmented_images


def create_augmented_array(images):
    augmented_images = augment_images_in_sequence(images)

    # Flatten all augmented images into a single list
    all_images = []
    for key in augmented_images.keys():
        all_images.extend(augmented_images[key])

    # Resize all images to (256, 256) if they are not already
    resized_images = [cv2.resize(img, (256, 256)) for img in all_images]

    # Convert the list to a numpy array
    augmented_array = np.array(resized_images)

    return augmented_array


def covariance_matrix(self, X):
    means = np.array([np.mean(X, axis=1)]).transpose()
    dev_matrix = X - means
    res = np.dot(dev_matrix, dev_matrix.transpose()) / (X.shape[1] - 1)
    return res


def plot_matrix_with_condition_number(matrix):
    """
    Plot a matrix with its condition number.

    Parameters:
    matrix (numpy.ndarray): The matrix to plot.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    M = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Matrix Plot")
    fig.colorbar(M)

    # Calculate and print the condition number
    cond_number = np.linalg.cond(matrix)
    print(f"The condition number of the matrix is: {cond_number}")

    plt.show()


def modify_image_array(images_array, replace_image, append_image):
    """
    Modify the image array by:
    1. Replacing the last image.
    2. Deleting the first image.
    3. Appending a new image at the end.

    Parameters:
    - images_array: numpy array of shape (4, 256, 256)
    - replace_image: numpy array of shape (256, 256), to replace the last image
    - append_image: numpy array of shape (256, 256), to append at the end

    Returns:
    - Modified numpy array
    """
    if images_array.shape != (4, 256, 256):
        raise ValueError("Original images array must have shape (4, 256, 256)")
    if replace_image.shape != (256, 256):
        raise ValueError("Replace image must have shape (256, 256)")
    if append_image.shape != (256, 256):
        raise ValueError("Append image must have shape (256, 256)")

    # Step 1: Replace the last image
    images_array[-1] = replace_image

    # Step 2: Delete the first image
    modified_array = np.delete(images_array, 0, axis=0)

    # Step 3: Append the new image at the end
    modified_array = np.append(
        modified_array, np.expand_dims(append_image, axis=0), axis=0
    )

    return modified_array
