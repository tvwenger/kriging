# kriging
## Ordinary and universal kriging in N dimensions.

`kriging` is a basic implementation of
[kriging](https://en.wikipedia.org/wiki/Kriging), a method of
interpolation using Gaussian process regression. `kriging` supports
ordinary kriging and universal kriging (using a polynomial drift
term), and three semivariogram models: Gaussian, spherical, and
exponential.

In the presence of drift (a varying mean value across the data space),
the observed semivariogram can be biased (see Starks & Fang, 1982,
Mathematical Geology, 14, 4:
https://doi.org/10.1007/BF01032592). `kriging` attempts to remove this
bias by first removing a fitted drift polynomial term before
generating the semivariogram.

### Installation
Install directly from this repository:
```bash
pip install git+https://github.com/tvwenger/kriging.git
```

Or, clone the repository and:
```bash
python setup.py install
```

### Usage
```python
from kriging import kriging
data_interp, var_interp = kriging.kriging(
    coord_obs, data_obs, coord_interp, e_data_obs=None,
    model=model, deg=deg, nbins=nbins, bin_number=bin_number,
    plot=plot)
```

Arguments:

* `coord_obs` is the `NxM` scalar array of `N` Cartesian positions in
  `M` dimensions.

* `data_obs` is the `N`-length scalar array of the observed value at
  each position.

* `coord_interp` is the `LxM` scalar array of `L` Cartesian positions
  at which to calculate interpolated values.

* `e_data_obs` (optional) is the `N`-length scalar array of observed
  value uncertainties. This is used to weight the data when fitting
  the drift polynomial. If `None` (default), then use equal weights.

* `model` (optional) is the assumed semivariogram model. Available
  values are `gaussian` (default), `spherical`, and `exponential`.

* `deg` (optional) is the degree of the polynomial drift term. `deg=0`
  (default) is equivalent to ordinary kriging (no drift).

* `nbins` (optional) is the number of lag bins to use when generating
  the semivariogram. The default value is `6`.

* `bin_number` (optional) is a flag to set how the lag bins are
  spaced. The default value is `False`, which means that the lag bins
  have equal width covering the full range of observed lags.  If
  `True`, then each lag bin includes the same number of data.

* `plot` (optional) is a filename where the semivariogram plot is
  saved. By default (`None`), the plot is not saved.

Return values:

* `data_interp` is the `L`-length scalar array of interpolated values at
  each `coord_interp` position.

* `var_interp` is the `L`-length scalar array of variances at each
  `coord_interp` position.

### Example
```python
import numpy as np
import matplotlib.pyplot as plt
from kriging import kriging

# set a random seed for reproduciblity
np.random.seed(1234)

# generate some random data
num_data = 100
coord_obs = np.random.uniform(-10, 10, size=(num_data, 2))
# the mean value varies quadratically
data_obs = 10.0 + 0.1 * coord_obs.prod(axis=1)
# add some noise that increases radially
e_data_obs = 0.1 + 0.1*np.sqrt((coord_obs**2.0).sum(axis=1))
data_obs += e_data_obs * np.random.randn(num_data)

# interpolation grid
xgrid, ygrid = np.mgrid[-10:10:100j, -10:10:100j]
coord_interp = np.vstack((xgrid.flatten(), ygrid.flatten())).T

# universal kriging with linear drift term
data_interp, var_interp = kriging.kriging(
    coord_obs, data_obs, coord_interp, e_data_obs=e_data_obs,
    model='gaussian', deg=1, nbins=10, bin_number=True,
    plot='semivariogram.pdf')

# plot data on top of interpolated grid
fig, ax = plt.subplots()
cax = ax.imshow(
    data_interp.reshape(100, 100).T, origin='lower', interpolation='none',
    extent=[-10, 10, -10, 10], vmin=0, vmax=20)
ax.scatter(
    coord_obs[:, 0], coord_obs[:, 1], c=data_obs, edgecolor='k',
    vmin=0, vmax=20)
fig.colorbar(cax)
ax.set_aspect('equal')
fig.savefig('example.pdf')
plt.close(fig)
```
![Semivariogram](https://raw.githubusercontent.com/tvwenger/kriging/master/example/semivariogram.pdf)
![Example](https://raw.githubusercontent.com/tvwenger/kriging/master/example/example.pdf)

### Issues and Contributing

Please submit issues or pull requests via
[Github](https://github.com/tvwenger/kriging).

### License and Warranty

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
