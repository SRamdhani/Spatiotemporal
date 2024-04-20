from sklearn.metrics.pairwise import euclidean_distances
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Simulation:
    @staticmethod
    def covariance_structure(sigDel: float, matElem: np.ndarray, phidel: float) -> np.ndarray:
        return sigDel * np.exp(-1 * matElem / phidel)

    # Borrowed from https://fanwangecon.github.io/M4Econ/panel/timeseries/fs_autoregressive.html
    @staticmethod
    def ar1(it_T: int = 1000, fl_constant: float =0.02, fl_normal_sd: float =0.02,
            fl_persistence: float =0.95, fl_shk_bnds: float =3, bl_init: bool =True) -> np.ndarray:

        fl_init = fl_constant / (1 - fl_persistence)

        # Generate a normal shock vector
        it_draws = it_T
        np.random.seed(789)
        ar_fl_shocks = np.random.normal(loc=0, scale=fl_normal_sd, size=it_draws)

        # out of bounds indicators
        fl_shk_bds_lower = 0 - fl_normal_sd * fl_shk_bnds
        fl_shk_bds_upper = 0 + fl_normal_sd * fl_shk_bnds
        ar_bl_outofbounds = (ar_fl_shocks <= fl_shk_bds_lower) | (ar_fl_shocks >= fl_shk_bds_upper)

        ar_fl_shocks[ar_fl_shocks <= fl_shk_bds_lower] = fl_shk_bds_lower
        ar_fl_shocks[ar_fl_shocks >= fl_shk_bds_upper] = fl_shk_bds_upper

        # Initialize output array
        ar_fl_time_series = np.zeros(len(ar_fl_shocks))

        for i in range(len(ar_fl_shocks)):
            if i == 0:
                # initialize using the ean of the process
                ar_fl_time_series[i] = fl_constant / (1 - fl_persistence)
                if bl_init: ar_fl_time_series[i] = fl_init
            else:
                fl_ts_t = fl_constant + ar_fl_time_series[i - 1] * fl_persistence + ar_fl_shocks[i]
                ar_fl_time_series[i] = fl_ts_t

        return ar_fl_time_series

    @staticmethod
    def create_dataset(sim_num: int = 1, size: int = 200, seeds: list = 234) -> np.ndarray:
        np.random.seed(seeds)
        x = np.random.uniform(size=size, low=0, high=25)
        np.random.seed(seeds+1)
        y = np.random.uniform(size=size, low=0, high=25)
        sample_points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        mat = euclidean_distances(sample_points)
        matCov = Simulation.covariance_structure(1.5, mat, 10)
        means = np.zeros(mat.shape[0])
        np.random.seed(seeds+2)
        sim_data = np.random.multivariate_normal(np.zeros(mat.shape[0]), matCov, sim_num)
        return sim_data



