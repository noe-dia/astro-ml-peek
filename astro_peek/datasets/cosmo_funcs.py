import camb 
import numpy as np
import powerbox as pbox
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt


def compute_pk(camb_params, kmin = 1e-3, kmax = 10):
    pars = camb.set_params(**camb_params)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    results = camb.get_results(pars)
    k, z, pk = results.get_matter_power_spectrum(
    minkh=kmin, maxkh=kmax, npoints=300
    )
    Pk = pk[0]
    return k, Pk

def compute_density_contrast_slice(k, Pk, img_size = 256, fov = 1256, seed = 42, return_volume = False):
    pk_func = InterpolatedUnivariateSpline(k, Pk, ext=1)
    lnpb = pbox.LogNormalPowerBox(
        N=img_size,                     # Number of grid-points in the box
        dim=3,                     # 2D box
        pk = pk_func, # The power-spectrum
        boxlength = fov,           # Size of the box (sets the units of k in pk)
        seed = seed              # Use the same seed as our powerbox
    )
    delta_vol = lnpb.delta_x() 
    if return_volume:
        return delta_vol, delta_vol[0]
    return delta_vol[0] # just one slice for the volume

def compute_density_contrast_from_scratch(camb_params, kmin = 1e-3, kmax = 1e1, img_size = 256, fov = 1256, seed = 42, return_volume = True): 
    k, Pk = compute_pk(camb_params, kmin = kmin, kmax = kmax)
    delta = compute_density_contrast_slice(k, Pk, img_size = img_size, fov = fov, seed = seed, return_volume = return_volume)
    return delta




# Utility func for plotting a 3D volume
from matplotlib.colors import LogNorm
def plot_volumes(data_list, titles=None, figsize=(12, 4), cmap='RdBu', norm = None, levels = np.logspace(np.log10(1e-2), np.log10(1e2), 256)):
	"""
	Plot 3D volumes using contour surfaces on three faces for each item in data_list.

	Parameters:
	- data_list: list of 3D numpy arrays shaped (Nx, Ny, Nz)
	- titles: optional list of strings for subplot titles (same length as data_list)
	- figsize: tuple for matplotlib figure size

	Returns:
	- fig: matplotlib Figure
	- axes: list of Axes3D objects
	"""
	if titles is None:
		titles = [f"Volume {i}" for i in range(len(data_list))]

	fig = plt.figure(figsize=figsize)
	axes = []

	for i, data in enumerate(data_list):
		if data.ndim != 3:
			raise ValueError(f"Expected 3D volume, got shape {data.shape}")

		Nx, Ny, Nz = data.shape[0], data.shape[1], data.shape[2]
		X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz))

		kw = {
			# 'vmin': float(np.min(data)),
			# 'vmax': float(np.max(data)),
			'levels': levels,
			'cmap': cmap,
			'norm': norm,
			'extend': 'both' # This ensures values < levels.min() are colored black
		}

		ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
		axes.append(ax)

		_ = ax.contourf(
			X[:, :, 0], Y[:, :, 0], data[:, :, -1],
			zdir='z', offset=Z.max(), **kw
		)
		_ = ax.contourf(
			X[0, :, :], data[0, :, :], Z[0, :, :],
			zdir='y', offset=Y.min(), **kw
		)
		_ = ax.contourf(
			data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
			zdir='x', offset=X.max(), **kw
		)

		xmin, xmax = float(X.min()), float(X.max())
		ymin, ymax = float(Y.min()), float(Y.max())
		zmin, zmax = float(Z.min()), float(Z.max())
		ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

		# edges_kw = dict(color='1', linewidth=1, zorder=1e3)
		# ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
		# ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
		# ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

		ax.set_title(titles[i])

		ax.view_init(30, -30, 0)
		ax.set_box_aspect(None, zoom=1.)
		# Remove background panes for a cleaner look
		ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

		# # 2. Remove the grid lines
		# ax.grid(False)

		# # 3. Remove the axis lines (spines) and ticks if you want a "floating" box
		# ax.set_axis_off()

	fig.tight_layout()
	return fig, axes

# Utility functions to set up priors from a yaml and to sample it
from scipy.stats import loguniform, uniform
def instantiate_prior(cfg): 
    """
    Hard-coded for the A_s (I know, sry...)
    """
    priors = {}
    for key in cfg.keys():
        a, b = cfg[key]
        if key == "As": 
            priors[key] = loguniform(float(a), float(b)) 

        else: 
            priors[key] = uniform(float(a), float(b))
    return priors

def sample_prior(priors, num_samples = 1000): 
    """
    Sample the prior obtained from the func instantiate_priors    
    """
    return np.stack([priors[key].rvs(num_samples) for key in priors.keys()]).T