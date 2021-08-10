import numpy as np
from scipy.signal import find_peaks, argrelextrema
from sklearn.neighbors import KernelDensity

import io_utils as ply


class Sign_Detector:
    def __init__(self, bw=2.5):
        self._kde = KernelDensity(kernel='gaussian', bandwidth=bw)

        self.pole_val = None
        self.plane_val = None  # To be found
        self.r_threshold = None

    def fit_kde(self, data):
        self._kde.fit(data.reshape(-1, 1))

    def produce_density_arr(self, data, n_points=0, show=False):
        if n_points <= 0:
            n_points = len(data)
        dens_x = np.linspace(data.min(), data.max(), n_points).reshape(-1, 1)

        density = np.exp(self._kde.score_samples(dens_x))

        if show:
            ply.plot_fun(density, dens_x, title='Estimated density')

        return density, dens_x

    def detect_modes(self, data, density_arr, ):
        distance = 0.25 * (data.max() - data.min())  # Minimum distance is 25% from overall range
        peaks_idx = find_peaks(density_arr, distance=distance)[0]

        print(f'{len(peaks_idx)} modes are found.')

        if len(peaks_idx) != 2:
            print('Amount of modes does not match 2 - this is not a sign.')
            return 0

        # find_peaks scans from the lowest values to higher,
        # thus first value is for pole with lower R
        self.pole_val, self.plane_val = data[peaks_idx[0]], data[peaks_idx[1]]

        min_idx = argrelextrema(density_arr[peaks_idx[0]: peaks_idx[-1]], np.less)[0]
        self.r_threshold = data[peaks_idx[0] + min_idx].mean()
        print(self.pole_val, self.plane_val)

        return 1
