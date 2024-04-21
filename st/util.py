from dataclasses import dataclass, field
from typing import Tuple
import tensorflow as tf
import numpy as np
import sklearn

@dataclass(frozen=False, unsafe_hash=True)
class Utility:
    @staticmethod
    def add_time_to_spatial(ar_fl_time_series: np.ndarray,
                            sim_data_ex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        st_data_true = []
        st_data_noise = []

        for s in ar_fl_time_series:
            st_data_true.append(sim_data_ex * s)
            st_data_noise.append(
                sim_data_ex * s + np.random.normal(size=sim_data_ex.shape, loc=0, scale=np.std(sim_data_ex) / 50))

        st_data_true = np.squeeze(np.stack(st_data_true, 2))
        st_data_noise = np.squeeze(np.stack(st_data_noise, 2))
        return st_data_true, st_data_noise

    @staticmethod
    def check_pca(principalComponents: np.ndarray,
                  pca: sklearn.decomposition._pca.PCA,
                  num_comp: int = 5) -> np.ndarray:

        return principalComponents[:, 0:num_comp].dot(pca.components_[:num_comp, :]) + \
                pca.mean_.reshape(-1, pca.components_.shape[1])

    @staticmethod
    def create_training_data_tf(principalComponents: np.ndarray,
                                pca: sklearn.decomposition._pca.PCA,
                                sample_points: np.ndarray) -> Tuple[tf.data.Dataset, tf.Tensor]:

        train_dataset = tf.data.Dataset.from_tensor_slices((sample_points, pca.components_.T.astype(np.float32),
                                                            # principalComponents.astype(np.float32),
                                                            pca.mean_.astype(np.float32).reshape(-1, 1)))

        train_dataset = train_dataset.shuffle(buffer_size=100).batch(50)
        principalComponentsTensor = tf.convert_to_tensor(principalComponents.astype(np.float32))
        return train_dataset, principalComponentsTensor

    @staticmethod
    def get_true_recon_field(principalComponents: np.ndarray,
                             pca: sklearn.decomposition._pca.PCA,
                             model_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # True Field
        true = principalComponents.dot(pca.components_) + \
               pca.mean_.reshape(-1, pca.components_.shape[1])

        # Reconstructed Field
        recon = principalComponents.dot(model_predictions.T) + \
                pca.mean_.reshape(-1, pca.components_.shape[1])

        return true, recon