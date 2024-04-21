from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Plots:

    @staticmethod
    def basic_time_plot(ar_fl_time_series: np.ndarray,
                        xlabel: str = 'Time Index',
                        ylabel: str = 'AR1 Value',
                        title: str = 'AR1 Time Series Example') -> None:

        plt.figure(figsize=(6, 2))
        plt.plot(ar_fl_time_series)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    @staticmethod
    def true_vs_recon_time_series(true: np.ndarray,
                                  recon: np.ndarray,
                                  sample_points: np.ndarray,
                                  total_plots: int = 5) -> None:

        for i in range(total_plots):  # Maximum number of plots is the size variable.
            plt.plot(true[:, i])
            plt.plot(recon[:, i], 'r')
            plt.title('Sample Point: ' + str(sample_points[i]) + ' Time Series')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.show()

    @staticmethod
    def spatial_heatmap(true: np.ndarray,
                        recon: np.ndarray,
                        sample_points: np.ndarray,
                        loc: int = 0,
                        total_plots: int = 5) -> None:

        # Create Meshgrid
        checkpoint = sample_points[loc]

        x1 = np.linspace(0, 25, 1000)
        X, Y = np.meshgrid(x1, x1)
        lims = [x1.min(), x1.max(), x1.min(), x1.max()]

        for i in range(total_plots):
            print('i: ', i, 'true: ', true[i, loc], 'recon: ', recon[i, loc])

            ndpol_true = interpolate.LinearNDInterpolator(sample_points, true[i])
            ndpol_recon = interpolate.LinearNDInterpolator(sample_points, recon[i])

            V_true = ndpol_true(list(zip(X.ravel(), Y.ravel()))).reshape(X.shape)
            V_recon = ndpol_recon(list(zip(X.ravel(), Y.ravel()))).reshape(X.shape)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Heatmaps')
            ax1.imshow(V_true, extent=lims, origin='lower')
            ax1.axes.set_title('True')
            ax1.axes.set_xlabel('x')
            ax1.axes.set_ylabel('y')
            ax1.axes.set_aspect('equal')
            ax1.annotate('x', xy=checkpoint, xycoords='data', color='w')
            ax1.scatter(sample_points[:, 0], sample_points[:, 1], s=1, c='red', marker='o')

            ax2.imshow(V_recon, extent=lims, origin='lower')
            ax2.axes.set_title('Recon')
            ax2.axes.set_xlabel('x')
            ax2.axes.set_ylabel('y')
            ax2.axes.set_aspect('equal')
            ax2.annotate('x', xy=checkpoint, xycoords='data', color='w')
            ax2.scatter(sample_points[:, 0], sample_points[:, 1], s=1, c='red', marker='o')

            plt.show()