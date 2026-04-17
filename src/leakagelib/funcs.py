import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
from .settings import *

KERNEL_ZS = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0,-4, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]) / 4**2
KERNEL_QS = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0,-1, 0, 0, 0,-1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]) / 4**2
KERNEL_US = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0,-2, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0,-2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]) / 4**2
KERNEL_ZK = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0,-8, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0,-8, 0,20, 0,-8, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0,-8, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
]) / 4**4 / 4
KERNEL_QK = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0,-4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
   [-1, 0, 4, 0, 0, 0, 4, 0,-1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0,-4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
]) / 4**4 / 3
KERNEL_UK = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0,-2, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0,-2, 0,12,0,-12, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0,-12,0,12, 0,-2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0,-2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]) / 4**4 / 3
KERNEL_XK = np.array([
    [0, 0, 0, 0,-1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0,-8, 0, 6, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
   [-1, 0,-8, 0,12, 0,-8, 0,-1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0,-8, 0, 6, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0,-1, 0, 0, 0, 0],
]) / 4**4 / 3
KERNEL_YK = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0,-4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0,-4, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0,-4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0,-4, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]) / 4**4 / 3

def super_zoom(image, frac, force_odd=False):
    """
    Interpolate zoom when zooming in, integrate zoom when zooming out.
    Zooming is centered and recommended for general use.

    Parameters
    ----------
    frac : float
        Zoom fraction.
    force_odd : bool, optional
        If True, round the image dimensions to an odd number of pixels.
    """

    if np.abs(frac - 1) < 1e-5:
        return image
    if frac > 1:
        return interpolate_zoom(image, frac, force_odd)
    else:
        return integrate_zoom(image, frac, force_odd)

def interpolate_zoom(image, frac, force_odd=False):
    """
    Zoom by linearly interpolating between known pixels to compute new pixels.
    Most suitable for zooming in.

    Parameters
    ----------
    frac : float
        Zoom fraction.
    force_odd : bool, optional
        If True, round the image dimensions to an odd number of pixels.
    """

    assert(len(image.shape) == 2 and image.shape[0] == image.shape[1])
    original_size = image.shape[0]

    # choose an odd new_size
    new_size = image.shape[0] * frac
    if force_odd:
        new_size = 2 * int(np.round((new_size + 1) / 2)) - 1
    else:
        new_size = int(np.round(new_size))

    if new_size == original_size:
        return image
    original_edges = np.linspace(-1, 1, original_size + 1)
    original_line = (original_edges[1:] + original_edges[:-1]) / 2
    interpolator = RegularGridInterpolator((original_line, original_line), image, method="linear", bounds_error=False, fill_value=None)

    edge_lim = (float(new_size) / 2) / frac / (original_size / 2)
    new_edges = np.linspace(-edge_lim, edge_lim, new_size + 1)
    new_line = (new_edges[1:] + new_edges[:-1]) / 2
    xs, ys = np.meshgrid(new_line, new_line, indexing="ij")
    new_image = interpolator((xs, ys))
    return new_image

def integrate_zoom_unvectorized(image, frac, force_odd=False):
    """
    Zoom by integrating over the original pixels. Not vectorized and very slow.
    Use `integrate_zoom` instead for efficiency.

    Parameters
    ----------
    frac : float
        Zoom fraction.
    force_odd : bool, optional
        If True, round the image dimensions to an odd number of pixels.
    """
    old_size = image.shape[0]
    new_size = image.shape[0] * frac
    if force_odd:
        new_size = 2 * int(np.round((new_size + 1) / 2)) - 1
    else:
        new_size = int(np.round(new_size))
        
    edge_lim = (float(new_size) / 2) / frac / (old_size / 2)
    old_edges = np.linspace(-1, 1, old_size + 1)
    new_edges = np.linspace(-edge_lim, edge_lim, new_size + 1)
    all_edges = np.concatenate([old_edges, new_edges])
    all_edges = np.unique(all_edges)
    all_edges = np.sort(all_edges)

    old_pixel_area = (old_edges[1] - old_edges[0])**2
    new_image = np.zeros((new_size, new_size))

    for intersection_idx in range(len(all_edges)-1):
        for intersection_idy in range(len(all_edges)-1):
            pixel_width = all_edges[intersection_idx + 1] - all_edges[intersection_idx]
            pixel_height = all_edges[intersection_idy + 1] - all_edges[intersection_idy]
            area = (pixel_width * pixel_height) / old_pixel_area
            if area <= 0:
                continue
            pixel_center_x = (all_edges[intersection_idx + 1] + all_edges[intersection_idx]) / 2
            pixel_center_y = (all_edges[intersection_idy + 1] + all_edges[intersection_idy]) / 2
            old_idx = np.searchsorted(old_edges, pixel_center_x) - 1
            old_idy = np.searchsorted(old_edges, pixel_center_y) - 1
            new_idx = np.searchsorted(new_edges, pixel_center_x) - 1
            new_idy = np.searchsorted(new_edges, pixel_center_y) - 1
            if new_idx < 0 or new_idy < 0 or new_idx >= new_size or new_idy >= new_size:
                continue
            if old_idx < 0 or old_idy < 0 or old_idx >= old_size or old_idy >= old_size:
                continue
            new_image[new_idx, new_idy] += image[old_idx, old_idy] * area
    
    return new_image

def integrate_zoom(image, frac, force_odd=False):
    """
    Zoom by integrating over the original pixels. Most suitable for zooming out.

    Parameters
    ----------
    frac : float
        Zoom fraction.
    force_odd : bool, optional
        If True, round the image dimensions to an odd number of pixels.
    """
    old_size = image.shape[0]
    new_size = image.shape[0] * frac
    if force_odd:
        new_size = 2 * int(np.round((new_size + 1) / 2)) - 1
    else:
        new_size = int(np.round(new_size))
    edge_lim = (float(new_size) / 2) / frac / (old_size / 2)
    old_edges = np.linspace(-1, 1, old_size + 1)
    new_edges = np.linspace(-edge_lim, edge_lim, new_size + 1)
    all_edges = np.concatenate([old_edges, new_edges])
    all_edges = np.unique(all_edges)
    all_edges = np.sort(all_edges)

    old_pixel_area = (old_edges[1] - old_edges[0])**2
    new_image = np.zeros((new_size, new_size))

    intersection_xs, intersection_ys = np.meshgrid(all_edges, all_edges, indexing="ij")
    center_xs = (intersection_xs[1:,1:] + intersection_xs[:-1,:-1]) / 2
    center_ys = (intersection_ys[1:,1:] + intersection_ys[:-1,:-1]) / 2
    areas = ((intersection_xs[1:,1:] - intersection_xs[:-1,:-1]) * (intersection_ys[1:,1:] - intersection_ys[:-1,:-1])) / old_pixel_area
    
    old_idx = np.searchsorted(old_edges, center_xs) - 1
    old_idy = np.searchsorted(old_edges, center_ys) - 1
    new_idx = np.searchsorted(new_edges, center_xs) - 1
    new_idy = np.searchsorted(new_edges, center_ys) - 1

    out_of_bounds_mask = \
        (new_idx < 0) | (new_idy < 0) | (new_idx >= new_size) | (new_idy >= new_size) | \
        (old_idx < 0) | (old_idy < 0) | (old_idx >= old_size) | (old_idy >= old_size)
    
    areas = areas[~out_of_bounds_mask]
    new_idx = new_idx[~out_of_bounds_mask]
    new_idy = new_idy[~out_of_bounds_mask]
    old_idx = old_idx[~out_of_bounds_mask]
    old_idy = old_idy[~out_of_bounds_mask]

    np.add.at(new_image, (new_idx, new_idy), image[old_idx, old_idy] * areas)
    
    return new_image

def _convolve(src, kernel, fix_edges=True):
    """
    Convolve an image with a convolution kernel.

    Parameters
    ----------
    src : ndarray
        The source image.
    kernel : ndarray
        The convolution kernel.
    fix_edges : bool, optional
        If True, remove edge effects caused by non-zero-sum kernels. Default is True.

    Returns
    -------
    array-like
        Convolved image
    """

    convolved = convolve(src, kernel, mode="same")
    if fix_edges:
        flat = np.ones_like(src)
        convolved /= convolve(flat, kernel, mode="same")

    return convolved