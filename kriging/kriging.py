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
2021-03-10 Trey V. Wenger - v1.3: Add support for covariances
"""

import numpy as np
import itertools
from scipy.spatial.distance import cdist, squareform
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from . import models

_VERSION = "1.3"

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
    data_cov=None,
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
        data_cov :: (N, N) array of scalars
            The data covariance matrix. If supplied, the value of
            e_data_obs is ignored.
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
    if data_cov is not None and data_cov.ndim != 2:
        raise ValueError("Covariance matrix must have two dimensions")
    if data_cov is not None and data_cov.shape[0] != data_cov.shape[1]:
        raise ValueError("Covariance matrix must be square")
    if data_cov is not None and data_cov.shape[0] != data_obs.shape[0]:
        raise ValueError("Covariance matrix and input data array mismatched")
    if model not in _MODELS:
        raise ValueError("Unsupported variogram model")

    data_weights = False
    if data_cov is not None:
        data_weights = True
    else:
        # equal weights for data if no errors supplied
        data_cov = np.eye(len(data_obs))
        if e_data_obs is not None:
            data_weights = True
            data_cov = data_cov * e_data_obs ** 2.0

    # generate polynomial basis vectors
    def basis(size, i):
        vector = np.zeros(size, dtype=np.int)
        vector[i] = 1
        return vector

    num_data, num_dim = coord_obs.shape
    polynomial_basis = [basis(num_dim + 1, i) for i in range(num_dim + 1)]

    # generate polynomial design matrix
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
    num_powers = design.shape[1]

    # evalue data_cov^-1 (dot) design
    cov_dot_design = np.linalg.solve(data_cov, design)
    # evaluate data_cov^-1 (dot) data_obs
    cov_dot_data = np.linalg.solve(data_cov, data_obs)

    # polynomial coefficients
    poly_coeff = np.linalg.inv(design.T.dot(cov_dot_design)).dot(
        design.T.dot(cov_dot_data)
    )

    # remove polynomial drift
    data_obs_res = data_obs - design.dot(poly_coeff)

    # generate pairwise distance, pairwise squared difference,
    # Jacobian matrix, and the diagonal Hessian matrix dot data_cov
    num_pairs = len(data_obs) * (len(data_obs) - 1) // 2
    coord_obs_dist = np.zeros(num_pairs)
    data_sqdiff = np.zeros(num_pairs)
    data_sqdiff_jac = np.zeros((num_pairs, len(data_obs)))
    data_sqdiff_hess_cov_diag = np.zeros(num_pairs)
    for i, (a, b) in enumerate(
        itertools.combinations(range(len(data_obs)), 2)
    ):
        coord_obs_dist[i] = np.sum((coord_obs[a] - coord_obs[b]) ** 2.0)
        data_diff = data_obs_res[a] - data_obs_res[b]
        data_sqdiff[i] = data_diff ** 2.0
        data_sqdiff_jac[i, a] = 2.0 * data_diff
        data_sqdiff_jac[i, b] = -2.0 * data_diff
        data_sqdiff_hess_cov_diag[i] = 2.0 * (
            data_cov[a, a] + data_cov[b, b] - data_cov[a, b]
        )

    # pairwise squared difference covariance
    data_sqdiff_cov = data_sqdiff_jac.dot(data_cov.dot(data_sqdiff_jac.T))
    data_sqdiff_cov += 0.5 * np.diag(data_sqdiff_hess_cov_diag ** 2.0)

    # define lag bins
    dist_min = np.min(coord_obs_dist)
    dist_max = np.max(coord_obs_dist)
    if bin_number:
        # bins have equal number of points in each
        num_points = int(len(coord_obs_dist) / nbins)
        dist_sorted = np.sort(coord_obs_dist)
        bins = [dist_sorted[n * num_points] for n in range(nbins)]
    else:
        # bins have fixed size
        bin_width = (dist_max - dist_min) / nbins
        bins = [dist_min + n * bin_width for n in range(nbins)]
    bins.append(dist_max)

    # bin data
    weights = 1.0 / np.diag(data_sqdiff_cov)
    bin_pop = np.zeros(nbins)
    lag_mean = np.ones(nbins) * np.nan
    semivar = np.ones(nbins) * np.nan
    semivar_std = np.ones(nbins) * np.nan
    for n in range(nbins):
        # indicies of bin members
        idx = np.where(
            (coord_obs_dist >= bins[n]) & (coord_obs_dist <= bins[n + 1])
        )[0]
        bin_pop[n] = len(idx)
        if bin_pop[n] > 0:
            lag_mean[n] = np.average(coord_obs_dist[idx], weights=weights[idx])
            semivar[n] = 0.5 * np.average(
                data_sqdiff[idx], weights=weights[idx]
            )
            semivar_std[n] = 0.5 * np.sqrt(
                0.5
                * np.sum(
                    [
                        1.0 / weights[i]
                        + 1.0 / weights[j]
                        + (data_sqdiff_cov[i, j] / weights[i] * weights[j])
                        for i in idx
                        for j in idx
                    ]
                )
            )

    # remove empty/nan bins
    bad = np.isnan(semivar)
    lag_mean = lag_mean[~bad]
    semivar = semivar[~bad]
    semivar_std = semivar_std[~bad]

    # fit semivariogram model to binned data
    semivariogram = _MODELS[model]

    # semivariogram soft_l1 loss function
    def loss(theta):
        res = (semivariogram(theta, lag_mean) - semivar) / semivar_std
        return np.sum(2.0 * np.sqrt(1.0 + res ** 2.0) - 1.0)

    p0 = [
        np.max(semivar) - np.min(semivar),
        np.median(lag_mean),
        semivar[0],
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
            semivar,
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

    # Generate the kriging weights matrix
    dist_mat = squareform(coord_obs_dist)
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
