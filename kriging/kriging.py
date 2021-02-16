"""
kriging.py

Ordinary and universal kriging in N dimensions.

Copyright(C) 2021 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

2021-02-15 Trey V. Wenger
"""

import numpy as np
import itertools
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from . import models

_VERSION = "1.2"

_MODELS = {
    "gaussian": models.gaussian,
    "exponential": models.exponential,
    "spherical": models.spherical,
}


def kriging(
    coord_obs,
    data_obs,
    coord_interp,
    e_data_obs=None,
    model="gaussian",
    deg=0,
    nbins=6,
    bin_number=False,
    plot=None,
):
    """
    Interpolation via N-dimensional Ordinary or Universal kriging.

    Inputs:
        coord_obs :: (N, M) array of scalars
            Cartesian coordinates of N observations in M dimensions
        data_obs :: (N,) array of scalars
            Observed data values
        coord_interp :: (L, M) array of scalars
            Cartesian coordinates at which to evaluate interpolation
        e_data_obs :: (N,) array of scalars
            Uncertainty of observed data values. If None, use equal
            data weights.
        model :: string
            Semivariogram model. One of the keys of _MODELS.
        deg :: integer
            Quadratic degree of local drift. If 0, then this is equivalent
            to ordinary kriging.
        nbins :: integer
            Number of lag bins
        bin_number :: boolean
            Define lag bins such that each bin includes the same number of data
            points. If False, use lag bins of equal width.
        plot :: string
            If not None, plot semivariogram and save to this filename.

    Returns: data_interp, var_interp
        data_interp :: (L,) array of scalars
            Interpolated data
        var_interp :: (L,) array of scalars
            Variance at each interpolation point
    """
    # Check inputs
    if coord_obs.ndim != 2:
        raise ValueError("Input coordinate array must be 2-D")
    if data_obs.ndim != 1:
        raise ValueError("Input data array must be 1-D")
    if coord_obs.shape[0] != data_obs.shape[0]:
        raise ValueError("Input coordinate array and data array mismatched")
    if e_data_obs is not None and e_data_obs.shape[0] != data_obs.shape[0]:
        raise ValueError("Input data array and e_data array mismatched")
    if model not in _MODELS:
        raise ValueError("Unsupported variogram model")

    # equal weights for data if no errors supplied
    if e_data_obs is None:
        e_data_obs = np.ones(len(data_obs))

    # generate polynomial basis vectors
    def basis(size, i):
        vector = np.zeros(size, dtype=np.int)
        vector[i] = 1
        return vector

    num_data, num_dim = coord_obs.shape
    polynomial_basis = [basis(num_dim + 1, i) for i in range(num_dim + 1)]

    # generate design matrix
    polynomial_powers = np.sum(
        list(itertools.combinations_with_replacement(polynomial_basis, deg)),
        axis=1,
    )
    coord_obs_pad = np.hstack(
        (np.ones((num_data, 1), dtype=coord_obs.dtype), coord_obs)
    )
    design = np.hstack(
        [
            (coord_obs_pad ** p).prod(axis=1)[..., None]
            for p in polynomial_powers
        ]
    )

    # polynomial soft_l1 loss function
    def loss(theta):
        res = (design.dot(theta) - data_obs) / e_data_obs
        return np.sum(2.0 * np.sqrt(1.0 + res ** 2.0) - 1.0)

    # fit polynomial drift
    num_powers = design.shape[1]
    theta0 = np.zeros(num_powers)
    # robust least squares
    res = minimize(loss, theta0)

    # remove polynomial drift
    data_obs_res = data_obs - design.dot(res.x)

    # evaluate pairwise distance between observations
    dist_obs = pdist(coord_obs, "euclidean")

    # evaluate pairwise squared difference between observations
    data_diff_obs = 0.5 * pdist(data_obs_res[:, np.newaxis], "sqeuclidean")

    # define lag bins
    dist_min = np.min(dist_obs)
    dist_max = np.max(dist_obs)
    if bin_number:
        # bins have equal number of points in each
        num_points = int(len(dist_obs) / nbins)
        dist_sorted = np.sort(dist_obs)
        bins = [dist_sorted[n * num_points] for n in range(nbins)]
    else:
        # bins have fixed size
        bin_width = (dist_max - dist_min) / nbins
        bins = [dist_min + n * bin_width for n in range(nbins)]
    bins.append(dist_max)

    # bin data
    bin_data = np.zeros(nbins)
    lag_mean = np.ones(nbins) * np.nan
    lag_std = np.ones(nbins) * np.nan
    semivar_mean = np.ones(nbins) * np.nan
    semivar_std = np.ones(nbins) * np.nan
    for n in range(nbins):
        members = (dist_obs >= bins[n]) * (dist_obs <= bins[n + 1])
        bin_data[n] = np.sum(members)
        if bin_data[n] > 0:
            lag_mean[n] = np.mean(dist_obs[members])
            lag_std[n] = np.std(dist_obs[members]) / np.sqrt(bin_data[n])
            semivar_mean[n] = np.mean(data_diff_obs[members])
            semivar_std[n] = np.std(data_diff_obs[members]) / np.sqrt(
                bin_data[n]
            )

    # remove empty/nan bins
    bad = np.isnan(semivar_mean)
    lag_mean = lag_mean[~bad]
    lag_std = lag_std[~bad]
    semivar_mean = semivar_mean[~bad]
    semivar_std = semivar_std[~bad]

    # fit semivariogram model to binned data
    semivariogram = _MODELS[model]

    # semivariogram soft_l1 loss function
    def loss(theta):
        res = (semivariogram(theta, lag_mean) - semivar_mean) / semivar_std
        return np.sum(2.0 * np.sqrt(1.0 + res ** 2.0) - 1.0)

    p0 = [
        np.max(semivar_mean) - np.min(semivar_mean),
        np.median(lag_mean),
        semivar_mean[0],
    ]
    bounds = [
        (0.0, np.inf),
        (np.min(lag_mean), np.max(lag_mean)),
        (0.0, np.inf),
    ]
    # robust least squares
    res = minimize(loss, p0, bounds=bounds, method="L-BFGS-B")

    # plot fitted semivariogram
    if plot is not None:
        xfit = np.linspace(0.0, 1.1 * lag_mean[-1], 100)
        yfit = semivariogram(res.x, xfit)
        fig, ax = plt.subplots()
        ax.errorbar(
            lag_mean,
            semivar_mean,
            yerr=semivar_std,
            fmt="o",
            color="k",
            linestyle="none",
        )
        ax.plot(xfit, yfit, "r-")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Semivariance")
        fig.tight_layout()
        fig.savefig(plot, bbox_inches="tight")
        plt.close(fig)

    # Generate the kriging matrix
    dist_mat = squareform(dist_obs)
    krig_mat = np.zeros((num_data + num_powers, num_data + num_powers))
    krig_mat[:num_data, :num_data] = semivariogram(res.x, dist_mat)
    krig_mat[num_data:, :num_data] = design.T
    krig_mat[:num_data, num_data:] = design
    np.fill_diagonal(krig_mat, 0.0)

    # Evaluate distance between each observation point and
    # each interpolation point, evaluate variogram
    num_interp = coord_interp.shape[0]
    interp_dist_mat = cdist(coord_interp, coord_obs, "euclidean")
    interp_variogram = np.zeros((num_interp, num_data + num_powers))
    interp_variogram[:, :num_data] = semivariogram(res.x, interp_dist_mat)
    coord_interp_pad = np.hstack(
        (np.ones((num_interp, 1), dtype=coord_interp.dtype), coord_interp)
    )
    interp_polynomial = np.hstack(
        [
            (coord_interp_pad ** p).prod(axis=1)[..., None]
            for p in polynomial_powers
        ]
    )
    interp_variogram[:, num_data:] = interp_polynomial

    # Solve kriging system of equations
    solution = np.linalg.solve(krig_mat, interp_variogram.T).T
    data_interp = np.sum(solution[:, :num_data] * data_obs, axis=1)
    var_interp = np.sum(solution * interp_variogram, axis=1)

    return data_interp, var_interp
