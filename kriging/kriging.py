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
"""

import numpy as np
import itertools
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import corner

from . import models

_VERSION = "2.0"

_MODELS = {
    "gaussian": models.gaussian,
    "exponential": models.exponential,
    "spherical": models.spherical,
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
        self.polynomial_powers = None
        self.polynomial_design = None
        self.num_powers = None
        self.model_params_pt = None

    def fit(
        self,
        model="gaussian",
        deg=0,
        nbins=6,
        bin_number=False,
        nsims=1000,
        corner_fname=None,
        semivariogram_fname=None,
    ):
        """
        Fit a polynomial drift and semivariogram model to the observed data.

        Inputs:
            model :: string
                Semivariogram model. One of the keys of _MODELS.
            deg :: integer
                Quadratic degree of local drift. If 0, then this is equivalent
                to ordinary kriging.
            nbins :: integer
                Number of lag bins
            bin_number :: boolean
                Define lag bins such that each bin includes the same number of
                data points. If False, use lag bins of equal width.
            nsims :: integer
                Number of Monte Carlo semivariogram model fitting simulations.
                If there are no observed data errors (i.e. if e_obs_data and
                obs_data_cov are None), then only one simulation is performed.
            corner_fname :: string
                If not None, plot the semivariogram model parameter corner
                plot and save to this filename
            semivariogram_fname :: string
                If not None, plot the semivariogram and save to this filename
        """
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
        poly_coeff = np.linalg.solve(
            self.polynomial_design.T.dot(cov_dot_design),
            self.polynomial_design.T.dot(cov_dot_data),
        )

        # remove polynomial drift
        obs_data_res = self.obs_data - self.polynomial_design.dot(poly_coeff)

        # define lag bin edges
        dist_min = np.min(self.obs_distance_pairs)
        dist_max = np.max(self.obs_distance_pairs)
        if bin_number:
            # bins have equal number of points in each
            num_points = int(len(self.obs_distance_pairs) / nbins)
            obs_distance_pairs_sorted = np.sort(self.obs_distance_pairs)
            bin_edges = [
                obs_distance_pairs_sorted[n * num_points] for n in range(nbins)
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

        # semivariogram model
        self.semivariogram = _MODELS[model]

        # semivariogram soft_l1 loss function
        def loss(theta, lag_mean, semivar):
            res = self.semivariogram(theta, lag_mean) - semivar
            return np.sum(2.0 * np.sqrt(1.0 + res ** 2.0) - 1.0)

        # semivariogram model parameter bounds
        bounds = [(0.0, np.inf), (0.0, np.inf), (0.0, np.inf)]

        # semivariogram model parameter estimates
        p0 = [0.0, np.median(lag_mean), 0.0]

        # Monte Carlo model parameters if supplied data errors
        resample = True
        if not self.has_obs_errors:
            nsims = 1
            resample = False
        model_params = np.zeros((nsims, 3))
        chol_cov = np.linalg.cholesky(self.obs_data_cov)
        semivars = np.zeros((nsims, nbins))

        for i in range(nsims):
            # resample data
            if resample:
                data_samples = obs_data_res + chol_cov.dot(
                    np.random.standard_normal(len(obs_data_res))
                )
            else:
                data_samples = obs_data_res

            # pairwise squared difference
            data_diff2 = (data_samples - data_samples[:, np.newaxis]) ** 2.0
            data_diff2 = data_diff2[self.upper]

            # compute semivariance
            for n in range(nbins):
                semivars[i, n] = np.mean(0.5 * data_diff2[bin_members[n]])

            # fit semivariogram model with robust least squares
            p0[0] = semivars[i].max() - semivars[i].min()
            p0[2] = semivars[i][0]
            res = minimize(
                loss,
                p0,
                args=(lag_mean, semivars[i]),
                bounds=bounds,
                method="L-BFGS-B",
            )
            model_params[i] = res.x

        # average fit parameter samples to get point estimate
        self.model_params_pt = np.mean(model_params, axis=0)

        # plot fitted semivariogram and parameter correlations
        if resample and corner_fname is not None:
            # generate corner plot
            fig = corner.corner(
                model_params,
                labels=["Sill", "Range", "Nugget"],
                truths=self.model_params_pt,
            )
            fig.savefig(corner_fname, bbox_inches="tight")
            plt.close(fig)

        # plot fitted semivariogram
        if semivariogram_fname is not None:
            xfit = np.linspace(0.0, 1.1 * lag_mean[-1], 100)
            yfit = self.semivariogram(self.model_params_pt, xfit)
            yfits = np.array(
                [self.semivariogram(params, xfit) for params in model_params]
            )
            yfits_lower = np.min(yfits, axis=0)
            yfits_upper = np.max(yfits, axis=0)
            fig, ax = plt.subplots()
            if resample:
                ax.violinplot(
                    semivars, positions=lag_mean, widths=1.0, showmeans=True
                )
                ax.fill_between(
                    xfit, yfits_lower, yfits_upper, color="r", alpha=0.2
                )
            else:
                ax.plot(lag_mean, semivars[0], "ko")
            ax.plot(xfit, yfit, "r-")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Semivariance")
            fig.tight_layout()
            fig.savefig(semivariogram_fname, bbox_inches="tight")
            plt.close(fig)

        # build kriging system of equations matrix
        self.krig_mat = np.zeros(
            (self.num_data + self.num_powers, self.num_data + self.num_powers)
        )
        self.krig_mat[: self.num_data, : self.num_data] = self.semivariogram(
            self.model_params_pt, self.obs_distance_full
        )
        if self.has_obs_errors:
            self.krig_mat[
                : self.num_data, : self.num_data
            ] += self.obs_data_cov
        self.krig_mat[
            self.num_data :, : self.num_data
        ] = self.polynomial_design.T
        self.krig_mat[
            : self.num_data, self.num_data :
        ] = self.polynomial_design
        if not self.has_obs_errors:
            np.fill_diagonal(self.krig_mat, 0.0)

    def interp(self, interp_pos):
        """
        Solve kriging system and evaluate interpolation and variance.

        Inputs:
            interp_pos :: (L, M) array of scalars
                Cartesian coordinates at which to evaluate interpolation

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
        num_interp = interp_pos.shape[0]

        # distance between each interpolation position and each
        # observation position
        interp_distance = np.sqrt(
            np.sum((interp_pos - self.obs_pos[:, np.newaxis]) ** 2.0, axis=-1)
        ).T

        # build interpolation matrix
        interp_mat = np.zeros((num_interp, self.num_data + self.num_powers))
        # evaluate variogram at each separation
        interp_mat[:, : self.num_data] = self.semivariogram(
            self.model_params_pt, interp_distance
        )
        if self.has_obs_errors:
            interp_mat[:, : self.num_data] += np.diag(self.obs_data_cov)
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

        # Solve kriging system of equations
        solution = np.linalg.solve(self.krig_mat, interp_mat.T).T
        data_interp = np.sum(
            solution[:, : self.num_data] * self.obs_data, axis=1
        )
        var_interp = np.sum(solution * interp_mat, axis=1)
        # catch small negative variances
        var_interp[var_interp < 0.0] = 0.0
        return data_interp, var_interp
