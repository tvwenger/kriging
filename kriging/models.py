"""
models.py

Semivariogram models.

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


def gaussian(theta, x):
    sill, range, nugget = theta
    return (
        sill * (1.0 - np.exp(-(x ** 2.0) / (range * 4.0 / 7.0) ** 2.0))
        + nugget
    )


def exponential(theta, x):
    sill, range, nugget = theta
    return sill * (1.0 - np.exp(-x / (range / 3.0))) + nugget


def spherical(theta, x):
    sill, range, nugget = theta
    ret = np.ones(x.shape, dtype=x.dtype) * (sill + nugget)
    c = x < range
    ret[c] = (
        sill
        * (1.0 - (1.0 - x[c] / (range / 3.0)) * np.exp(-x[c] / (range / 3.0)))
        + nugget
    )
    return ret
