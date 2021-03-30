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
2021-03-14 Trey V. Wenger - v2.0
    Proper handling of data uncertainties and covariances.
2021-03-30 Trey V. Wenger - v2.1
    Improve data uncertainty handling, rename "semivariogram"
    to "variogram" and "semivariance" to "gamma-variance" for
    clarity.
"""

import numpy as np
import itertools
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from . import models

_VERSION = "2.1"

_MODELS = {
    "linear": models.linear,
    "gaussian": models.gaussian,
    "exponential": models.exponential,
    "spherical": models.spherical,
    "wave": models.wave,
    "quadratic": models.quadratic,
    "circular": models.circular,
}


class Kriging:
    def __init__(self, obs_pos, obs_data, e_obs_data=None, obs_data_cov=None):
        """
        Initialize a new Kriging object.

        Inputs:
            obs_pos :: (N, M) array of scalars
                Cartesian coordinates of N observations in M dimensions
            obs_data :: (N,) array of scalars
                Observed data values
            e_obs_data :: (N,) array of scalars
                Uncertainty of observed data values. If None, use obs_data_cov
                or equal data weights.
            obs_data_cov :: (N, N) array of scalars
                The data covariance matrix. If supplied, the value of
                e_obs_data is ignored.

        Returns: Nothing
        """
        # Check inputs
        if obs_pos.ndim != 2:
            raise ValueError("Observed coordinate array must be 2-D")
        if obs_data.ndim != 1:
            raise ValueError("Observed data array must be 1-D")
        if obs_pos.shape[0] != obs_data.shape[0]:
            raise ValueError(
                "Observed coordinate array and data array mismatched"
            )
        if e_obs_data is not None and e_obs_data.shape[0] != obs_data.shape[0]:
            raise ValueError("Observed data array and e_data array mismatched")
        if obs_data_cov is not None and obs_data_cov.ndim != 2:
            raise ValueError("Covariance matrix must have two dimensions")
        if obs_data_cov is not None:
            if obs_data_cov.shape[0] != obs_data_cov.shape[1]:
                raise ValueError("Covariance matrix must be square")
            if obs_data_cov.shape[0] != obs_data.shape[0]:
                raise ValueError(
                    "Covariance matrix and input data array mismatched"
                )

        self.obs_pos = obs_pos
        self.obs_data = obs_data
        self.num_data, self.num_dim = self.obs_pos.shape

        # build covariance matrix
        self.has_obs_errors = False
        if obs_data_cov is None:
            # equal weights for data if no errors supplied
            self.obs_data_cov = np.eye(len(obs_data))
            if e_obs_data is not None:
                self.obs_data_cov = self.obs_data_cov * e_obs_data ** 2.0
                self.has_obs_errors = True
        else:
            self.obs_data_cov = obs_data_cov
            self.has_obs_errors = True

        # pairwise distance between each observation
        self.upper = np.triu_indices(len(self.obs_data), k=1)
        self.obs_distance_full = np.sqrt(
            np.sum(
                (self.obs_pos - self.obs_pos[:, np.newaxis]) ** 2.0, axis=-1
            )
        )
        self.obs_distance_pairs = self.obs_distance_full[self.upper]

        # attributes filled by fit function
        self.poly_coeff = None
        self.polynomial_powers = None
        self.polynomial_design = None
        self.num_powers = None
        self.model_params = None

    def fit(
        self,
        model="gaussian",
        deg=0,
        nbins=10,
        bin_number=False,
        lag_cutoff=1.0,
    ):
        """
        Fit a polynomial drift and variogram model to the observed data.

        Inputs:
            model :: string
                Variogram model. One of the keys of _MODELS.
            deg :: integer
                Quadratic degree of local drift. If 0, then this is equivalent
                to ordinary kriging.
            nbins :: integer
                Number of lag bins.
            bin_number :: boolean
                Define lag bins such that each bin includes the same number of
                data points. If False, use lag bins of equal width.
            lag_cutoff :: scalar between 0.0 and 1.0
                Ignore points separated by more than this fraction of the
                maximum lag when plotting and fitting the variogram.

        Returns: variogram_fig
            variogram_fig :: matplotlib.pyplot.Figure
                The variogram plot
        """
        # check inputs
        if deg < 0:
            raise ValueError("deg must be >= 0")
        if nbins < 3:
            raise ValueError("nbins must be >= 3")
        if lag_cutoff <= 0.0 or lag_cutoff > 1.0:
            raise ValueError("lag_cutoff must be in (0.0, 1.0]")

        # generate polynomial basis vectors
        def basis(size, i):
            vector = np.zeros(size, dtype=np.int)
            vector[i] = 1
            return vector

        polynomial_basis = [
            basis(self.num_dim + 1, i) for i in range(self.num_dim + 1)
        ]

        # generate polynomial design matrix
        self.polynomial_powers = np.sum(
            list(
                itertools.combinations_with_replacement(polynomial_basis, deg)
            ),
            axis=1,
        )
        obs_pos_pad = np.hstack(
            (
                np.ones((self.num_data, 1), dtype=self.obs_pos.dtype),
                self.obs_pos,
            )
        )
        self.polynomial_design = np.hstack(
            [
                (obs_pos_pad ** p).prod(axis=1)[..., None]
                for p in self.polynomial_powers
            ]
        )
        self.num_powers = self.polynomial_design.shape[1]

        # polynomial coefficients via generalized least squares
        cov_dot_design = np.linalg.solve(
            self.obs_data_cov, self.polynomial_design
        )
        cov_dot_data = np.linalg.solve(self.obs_data_cov, self.obs_data)
        self.poly_coeff = np.linalg.solve(
            self.polynomial_design.T.dot(cov_dot_design),
            self.polynomial_design.T.dot(cov_dot_data),
        )

        # remove polynomial drift
        obs_data_res = self.obs_data - self.polynomial_design.dot(
            self.poly_coeff
        )

        # define lag bin edges
        dist_min = np.min(self.obs_distance_pairs)
        dist_max = lag_cutoff * np.max(self.obs_distance_pairs)
        if bin_number:
            # bins have equal number of points in each
            num_good_points = int(np.sum(self.obs_distance_pairs <= dist_max))
            num_bin_points = int(num_good_points / nbins)
            obs_distance_pairs_sorted = np.sort(self.obs_distance_pairs)
            bin_edges = [
                obs_distance_pairs_sorted[n * num_bin_points]
                for n in range(nbins)
            ]
        else:
            # bins have fixed size
            bin_width = (dist_max - dist_min) / nbins
            bin_edges = [dist_min + n * bin_width for n in range(nbins)]
        bin_edges.append(dist_max)

        # assign data pairs to bins
        lag_mean = np.zeros(nbins)
        bin_members = []
        for n in range(nbins):
            idx = np.where(
                (self.obs_distance_pairs >= bin_edges[n])
                & (self.obs_distance_pairs <= bin_edges[n + 1])
            )[0]
            lag_mean[n] = np.mean(self.obs_distance_pairs[idx])
            bin_members.append(idx)

        # pairwise squared difference
        data_diff2 = (obs_data_res - obs_data_res[:, np.newaxis]) ** 2.0
        data_diff2 = data_diff2[self.upper]

        # compute gamma-variance
        gammavars = np.ones(nbins) * np.nan
        for n in range(nbins):
            gammavars[n] = 0.5 * np.mean(data_diff2[bin_members[n]])

        # variogram model
        self.variogram = _MODELS[model]

        # variogram soft_l1 loss function
        def loss(theta, lag_mean, gammavar):
            res = self.variogram(theta, lag_mean) - gammavar
            return np.sum(2.0 * np.sqrt(1.0 + res ** 2.0) - 1.0)

        # variogram model parameter bounds
        bounds = [
            (0.0, np.inf),
            (self.obs_distance_pairs.min(), np.inf),
            (0.0, np.inf),
        ]

        # variogram model parameter estimates
        p0 = [
            gammavars.max() - gammavars.min(),
            np.median(lag_mean),
            gammavars[0],
        ]

        # fit variogram model with robust least squares
        res = minimize(
            loss,
            p0,
            args=(lag_mean, gammavars),
            bounds=bounds,
            method="L-BFGS-B",
        )
        self.model_params = res.x

        # plot fitted variogram
        xfit = np.linspace(0.0, 1.1 * lag_mean[-1], 100)
        yfit = self.variogram(self.model_params, xfit)
        variogram_fig, ax = plt.subplots()
        ax.plot(lag_mean, gammavars, "ko")
        ax.plot(xfit, yfit, "r-", linewidth=2.0)
        ax.set_xlabel("Lag")
        ax.set_ylabel(r"$\gamma$-variance")
        variogram_fig.tight_layout()

        # build kriging system of equations matrix
        self.krig_mat = np.zeros(
            (self.num_data + self.num_powers, self.num_data + self.num_powers)
        )
        self.krig_mat[: self.num_data, : self.num_data] = self.variogram(
            self.model_params, self.obs_distance_full
        )
        self.krig_mat[
            self.num_data :, : self.num_data
        ] = self.polynomial_design.T
        self.krig_mat[
            : self.num_data, self.num_data :
        ] = self.polynomial_design
        np.fill_diagonal(self.krig_mat, 0.0)
        if self.has_obs_errors:
            self.krig_mat[
                : self.num_data, : self.num_data
            ] -= self.obs_data_cov
        return variogram_fig

    def interp(self, interp_pos, resample=False):
        """
        Solve kriging system and evaluate interpolation and variance.

        Inputs:
            interp_pos :: (L, M) array of scalars
                Cartesian coordinates at which to evaluate interpolation
            resample :: boolean
                If True, generate a resampled interpolation map
                by sampling the observed data

        Returns: interp_data, interp_var
            interp_data :: (L,) array of scalars
                Interpolated at each interpolation point
            interp_var :: (L,) array of scalars
                Variance at each interpolation point
        """
        # check inputs
        if interp_pos.shape[1] != self.num_dim:
            raise ValueError(
                "Interpolation coordinates have incorrect dimensions"
            )
        if self.krig_mat is None:
            raise ValueError("Run fit function first")
        if resample and not self.has_obs_errors:
            raise ValueError("Observed data uncertainties not provided")
        num_interp = interp_pos.shape[0]

        # distance between each interpolation position and each
        # observation position
        interp_distance = np.sqrt(
            np.sum((interp_pos - self.obs_pos[:, np.newaxis]) ** 2.0, axis=-1)
        ).T

        # build interpolation matrix
        interp_mat = np.zeros((num_interp, self.num_data + self.num_powers))
        # evaluate variogram at each separation
        interp_mat[:, : self.num_data] = self.variogram(
            self.model_params, interp_distance
        )
        # if self.has_obs_errors:
        #    interp_mat[:, : self.num_data] += np.diag(self.obs_data_cov)
        # evaluate polynomial basis at each separation
        interp_pos_pad = np.hstack(
            (np.ones((num_interp, 1), dtype=interp_pos.dtype), interp_pos)
        )
        interp_polynomial_design = np.hstack(
            [
                (interp_pos_pad ** p).prod(axis=1)[..., None]
                for p in self.polynomial_powers
            ]
        )
        interp_mat[:, self.num_data :] = interp_polynomial_design

        # resample
        if resample:
            chol_cov = np.linalg.cholesky(self.obs_data_cov)
            obs_data = self.obs_data + chol_cov.dot(
                np.random.standard_normal(len(self.obs_data))
            )
        else:
            obs_data = self.obs_data

        # Solve kriging system of equations
        solution = np.linalg.solve(self.krig_mat, interp_mat.T).T
        data_interp = np.sum(solution[:, : self.num_data] * obs_data, axis=1)
        var_interp = np.sum(solution * interp_mat, axis=1)
        return data_interp, var_interp
