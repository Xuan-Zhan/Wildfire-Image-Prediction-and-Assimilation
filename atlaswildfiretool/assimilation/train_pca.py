import numpy as np
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt


class TrainPCA:
    """
    This class is designed to train a Principal Component Analysis (PCA) model using a
    specified dataset and apply the trained PCA to reduce dimensionality of observation
    and background datasets. It then calculates and compares the Mean Squared Error (MSE)
    between original and reduced data spaces to assess performance.

    Attributes:
        train_path (str): Path to the numpy file containing data used for training the PCA.
        obs_path (str): Path to the numpy file containing observation data to test the PCA's
                        performance by comparing with model predictions.
        background_path (str): Path to the numpy file containing background data to test the
                        PCA's performance by comparing with actual observations.
        n_components (int): The number of principal components to retain in the PCA, determining
                        the size of the latent space. Default is 256.
        obs_index (int): Index of the specific observation image from the dataset to be used.
                        Default is -1, indicating the last image.
        background_index (int): Index of the specific background image from the dataset to be used.
                        Default is -1, indicating the last image.

    The class automatically:
        - Loads the data from the specified paths.
        - Flattens the images to fit PCA requirements.
        - Trains the PCA model on the training dataset.
        - Applies the PCA transformation to both the observation and background images.
        - Calculates the MSE in both original and PCA-reduced data spaces.
        - Allows for saving the trained PCA model and decompressing data for visual assessment.
    """

    def __init__(
        self,
        train_path,
        obs_path,
        background_path,
        n_components=256,
        obs_index=-1,
        background_index=-1,
    ):
        # load data
        self.train_data = np.load(train_path)
        self.obs = np.load(obs_path)
        self.background = np.load(background_path)
        self.obs_index = obs_index
        self.background_index = background_index

        # flat data
        self.train_data_flat = self.train_data.reshape(self.train_data.shape[0], -1)
        self.obs_flat = self.obs[self.obs_index].reshape(1, -1)
        self.background_flat = self.background[self.background_index].reshape(1, -1)

        # train PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.train_data_flat)

        # reduce data
        self.obs_reduced = self.pca.transform(self.obs_flat)
        self.background_reduced = self.pca.transform(self.background_flat)

        # calculate MSE
        self.mse_original = np.mean((self.obs_flat - self.background_flat) ** 2)
        self.mse_reduced = np.mean((self.obs_reduced - self.background_reduced) ** 2)

    def print_mse(self):
        print(
            "MSE between the ",
            self.obs_index,
            "th image of obsevation and background dataset in physic space:",
            self.mse_original,
        )
        print(
            "MSE between the ",
            self.background_index,
            "th image of obsevation and background dataset in physic space:",
            self.mse_reduced,
        )

    def save_model(self, filename="pca_model.pkl"):
        joblib.dump(self.pca, filename)

    def get_reduced_data(self):
        return self.obs_reduced, self.background_reduced

    def decompress_data(self):
        # decompress data
        self.obs_decompressed = self.pca.inverse_transform(self.obs_reduced).reshape(
            256, 256
        )
        self.background_decompressed = self.pca.inverse_transform(
            self.background_reduced
        ).reshape(256, 256)
        return self.obs_decompressed, self.background_decompressed

    def plot_images(self):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(self.obs[self.obs_index].reshape(256, 256), cmap="gray")
        ax[0, 0].set_title("Original Observation")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(
            self.background[self.background_index].reshape(256, 256), cmap="gray"
        )
        ax[0, 1].set_title("Original Background")
        ax[0, 1].axis("off")

        self.decompress_data()  # decompress data

        ax[1, 0].imshow(self.obs_decompressed, cmap="gray")
        ax[1, 0].set_title("Decompressed Observation")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(self.background_decompressed, cmap="gray")
        ax[1, 1].set_title("Decompressed Background")
        ax[1, 1].axis("off")

        plt.show()
