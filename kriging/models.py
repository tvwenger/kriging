"""
models.py

Variogram models.

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
2021-03-20 Trey V. Wenger - added linear, wave, quadratic, circular models
2021-04-03 Trey V. Wenger - optimizations using numpy.piecewise and numpy.sinc
"""

import numpy as np


def linear(theta, x):
    sill, range, nugget = theta
    return np.piecewise(
        x,
        [x < range, x >= range],
        [lambda x: nugget + sill / range * x, nugget + sill],
    )


def gaussian(theta, x):
    sill, range, nugget = theta
    return sill * (1.0 - np.exp(-5.0 * ((x / range) ** 2.0))) + nugget


def exponential(theta, x):
    sill, range, nugget = theta
    return sill * (1.0 - np.exp(-3.0 * x / range)) + nugget


def spherical(theta, x):
    sill, range, nugget = theta
    return np.piecewise(
        x,
        [x < range, x >= range],
        [
            lambda x: sill
            * ((3.0 * x) / (2.0 * range) - (x ** 3.0) / (2.0 * range ** 3.0))
            + nugget,
            sill + nugget,
        ],
    )


def wave(theta, x):
    sill, range, nugget = theta
    return sill * (1.0 - np.sinc(3.0 * x / range / np.pi)) + nugget


def quadratic(theta, x):
    sill, range, nugget = theta
    return (
        sill * ((3.0 * x / range) ** 2.0 / (1.0 + (3.0 * x / range) ** 2.0))
        + nugget
    )


def circular(theta, x):
    sill, range, nugget = theta
    return np.piecewise(
        x,
        [x < range, x >= range],
        [
            lambda x: sill
            * (
                1.0
                - 2.0 / np.pi * np.arccos(x / range)
                + 2.0 * x / (np.pi * range) * np.sqrt(1.0 - (x / range) ** 2.0)
            )
            + nugget,
            sill + nugget,
        ],
    )
