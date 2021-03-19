# kriging (v2.0)
### Ordinary and universal kriging in N dimensions.

`kriging` is a basic implementation of
[kriging](https://en.wikipedia.org/wiki/Kriging), a method of
interpolation using Gaussian process regression. `kriging` supports
ordinary kriging and universal kriging (using a polynomial drift
term), and three semivariogram models: Gaussian, spherical, and
exponential.

In the presence of drift (a varying mean value across the data space),
the observed semivariogram can be biased (see Starks & Fang, 1982,
Mathematical Geology, 14, 4;
https://doi.org/10.1007/BF01032592). `kriging` attempts to remove this
bias by subtracting a fitted polynomial drift term before generating
the semivariogram.

`kriging` also handles data uncertainties and covariances. If the
data have associated errors, then the semivariogram model is derived
from many Monte Carlo realizations of the data. The data variances and
covariances are also factored into the kriging system of equations so that
the interpolation variances reflect the observed data uncertainties (see,
for example, Cecinati et al., 2018, Atmosphere, 9(11), 446; https://doi.org/10.3390/atmos9110446).

## Installation
Install directly from this repository:
```bash
pip install git+https://github.com/tvwenger/kriging.git
```

Or, clone the repository and:
```bash
python setup.py install
```

## Usage
```python
from kriging import kriging
krig = kriging.Kriging(obs_pos, obs_data, e_obs_data=e_obs_data, obs_data_cov=obs_data_cov)
krig.fit(model=model, deg=deg, nbins=nbins, bin_number=bin_number, nsims=nsims,
         corner_fname=corner_fname, semivariogram_fname=semivariogram_fname)
interp_data, interp_var = krig.interp(interp_pos)
```

## Functions & Arguments:

### Object initialization
Intialize a new `Kriging` object.
```python
krig = kriging.Kriging(obs_pos, obs_data, e_obs_data=e_obs_data, obs_data_cov=obs_data_cov)
```
* `obs_pos` is the `NxM` scalar array of the `N` observed Cartesian positions in
  `M` dimensions.
* `obs_data` is the `N`-length scalar array of observations at
  each position.
* `e_obs_data` (optional) is the `N`-length scalar array of observed
  value uncertainties (standard deviation). If `None` (default), then the
  data covariance matrix can be supplied via `obs_data_cov`. If both
  are `None`, then data uncertainies are not considered in the kriging solution.
* `obs_data_cov` (optional) is the `NxN` scalar array of observed
  data covariances.  If `None` (default), then the (uncorrelated)
  data uncertainties can be supplied via `e_obs_data`. If both
  are `None`, then data uncertainies are not considered in the kriging solution.

### Fitting Semivariogram Model
Fit and remove a polynomial drift component and then fit a semivariogram model to the
drift-subtracted data.
```python
krig.fit(model=model, deg=deg, nbins=nbins, bin_number=bin_number, nsims=nsims,
         corner_fname=corner_fname, semivariogram_fname=semivariogram_fname)
```
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
* `nsims` (optional) is the number of Monte Carlo simulations to perform
  when fitting the semivariogram model. The default is `1000`. This
  parameter is silently ignored if both `e_obs_data` and `obs_data_cov` are `None`.
* `corner_fname` (optional) is the filename (including extension) for the
  semivariogram model parameter corner plot. If `None`, then no figure is created.
  This plot shows the marginalized distributions of the semivariogram model parameters
  determined from the Monte Carlo simulations. The default is `None`, which will not produce
  a figure. This parameter is silently ignored if both `e_obs_data` and
  `obs_data_cov` are `None`.
* `semivariogram_fname` (optional) is the filename (including extension) for the
  fitted semivariogram model plot. If `None`, then no figure is created. 
  If both `e_obs_data` and `obs_data_cov` are `None`, then this plot shows the semivariogram
  and fitted semivariogram model. Otherwise, this plot shows the Monte Carlo distribution of
  semivariogram bin mean values as a violin plot, the range of all fitted semivariogram
  models, and the average semivariogram model.

### Interpolation
Solve the kriging system of equations and evaluate the interpolation and
variance a given positions.
```python
interp_data, interp_var = krig.interp(interp_pos)
```
* `interp_pos` is the `LxM` scalar array of `L` Cartesian positions
  at which to calculate interpolated values.
* `interp_data` is the `L`-length scalar array of interpolated values at
  each `interp_pos` position.
* `interp_var` is the `L`-length scalar array of variances at each
  `interp_pos` position.

## Examples
### Ordinary Kriging
```python
import numpy as np
import matplotlib.pyplot as plt
from kriging import kriging

# set a random seed for reproduciblity
np.random.seed(1234)

# generate some random data
num_data = 100
obs_pos = np.random.uniform(-10, 10, size=(num_data, 2))
# the mean value varies quadratically
obs_data = 10.0 + 0.1 * obs_pos.prod(axis=1)

# plot function
def plot(data, fname, color=None, vmin=None, vmax=None, label=None):
    fig, ax = plt.subplots()
    cax = ax.imshow(
        data.reshape(120, 120).T, origin="lower", interpolation="none",
        extent=[-12, 12, -12, 12], vmin=vmin, vmax=vmax)
    ax.scatter(
        obs_pos[:, 0], obs_pos[:, 1], c=color, edgecolor="k",
        vmin=vmin, vmax=vmax)
    fig.colorbar(cax, label=label)
    ax.set_aspect("equal")
    fig.savefig(fname)
    plt.close(fig)

# interpolation grid
xgrid, ygrid = np.mgrid[-12:12:120j, -12:12:120j]
interp_pos = np.vstack((xgrid.flatten(), ygrid.flatten())).T

# evaluate "true" field, plot
true_data = 10.0 + 0.1 * interp_pos.prod(axis=1)
plot(true_data, "example/truth.png", color='k',
     vmin=0.0, vmax=25.0, label="Truth")
```
<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/truth.png" width="45%" />
```python
# ordinary kriging
krig = kriging.Kriging(obs_pos, obs_data)
krig.fit(model="gaussian", nbins=10, bin_number=True,
         semivariogram_fname="example/semivariogram_ordinary.png")
interp_data, interp_var = krig.interp(interp_pos)

# plot data on top of interpolated grid
plot(interp_data, "example/interp_ordinary.png", color=obs_data,
     vmin=0.0, vmax=25.0, label="Interpolation")
plot(np.sqrt(interp_var), "example/std_ordinary.png", color='k',
     vmin=None, vmax=None, label="Standard Deviation")

# plot difference between interpolated and true grid
plot(true_data - interp_data, "example/diff_ordinary.png", color='k',
     vmin=-0.2, vmax=0.2, label="Truth $-$ Interp")
```
<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/semivariogram_ordinary.png" width="45%" /><img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/interp_ordinary.png" width="45%" />
<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/std_ordinary.png" width="45%" /><img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/diff_ordinary.png" width="45%" />

### Ordinary Kriging with Constant Noise
```python
# add constant random noise
e_obs_data = 0.1 * np.ones(len(obs_data))
obs_noisy_data = obs_data + e_obs_data * np.random.randn(len(obs_data))

# ordinary kriging
krig = kriging.Kriging(obs_pos, obs_noisy_data, e_obs_data=e_obs_data)
krig.fit(model="gaussian", deg=1, nbins=10, bin_number=True, nsims=10000,
         corner_fname="example/corner_ordinary_noise.png",
         semivariogram_fname="example/semivariogram_ordinary_noise.png")
interp_data, interp_var = krig.interp(interp_pos)

# plot data on top of interpolated grid
plot(interp_data, "example/interp_ordinary_noise.png", color=obs_noisy_data,
     vmin=0.0, vmax=25.0, label="Interpolation")
plot(np.sqrt(interp_var), "example/std_ordinary_noise.png", color='k',
     vmin=None, vmax=None, label="Standard Deviation")

# plot difference between interpolated and true grid
plot(true_data - interp_data, "example/diff_ordinary_noise.png", color='k',
     vmin=None, vmax=None, label="Truth $-$ Interp")
```
<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/corner_ordinary_noise.png" width="45%" />
<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/semivariogram_ordinary_noise.png" width="45%" /><img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/interp_ordinary_noise.png" width="45%" />
<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/std_ordinary_noise.png" width="45%" /><img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/diff_ordinary_noise.png" width="45%" />
```python
# universal kriging with linear drift, accounting for error
krig = kriging.Kriging(obs_pos, obs_data, e_obs_data=e_obs_data)
krig.fit(model="gaussian", deg=1, nbins=10, bin_number=True, nsims=10000,
         corner_fname="example/corner_universal_noise.png",
         semivariogram_fname="example/semivariogram_universal_noise.png")
interp_data, interp_var = krig.interp(interp_pos)

# plot data on top of interpolated grid
plot(interp_data, "example/interp_universal_noise.png", color=obs_data,
     vmin=0.0, vmax=25.0, label="Interpolation")
plot(np.sqrt(interp_var), "example/std_universal_noise.png", color='k',
     vmin=None, vmax=None, label="Standard Deviation")

# plot difference between interpolated and true grid
plot(true_data - interp_data, "example/diff_universal_noise.png", color='k',
     vmin=None, vmax=None, label="Truth $-$ Interp")

# add some noise that increases radially
e_obs_data += 0.1*np.sqrt((obs_pos**2.0).sum(axis=1))
obs_data += e_obs_data * np.random.randn(num_data)
```






```
<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/corner.png" width="45%" /><img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/semivariogram.png" width="45%" />

<img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/example.png" width="45%" /><img src="https://raw.githubusercontent.com/tvwenger/kriging/master/example/example_std.png" width="45%" />

## Issues and Contributing

Please submit issues or pull requests via
[Github](https://github.com/tvwenger/kriging).

## License and Warranty

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
