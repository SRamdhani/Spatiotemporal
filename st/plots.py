from dataclasses import dataclass, field
import matplotlib.pyplot as plt
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