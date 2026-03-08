import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import convolve
from scipy.interpolate import interp1d, RegularGridInterpolator
import leakagelib

IXPE_PIXEL_SIZE = 2.6
PIXEL_WIDTH = 2.9729

def blur(image, sigma):
    # Blur the image where sigma is in units of pixels
    if sigma == 0: return image
    width = int(np.ceil(3 * sigma))
    line = np.arange(-width, width+1)
    xs, ys = np.meshgrid(line, line)
    gauss = np.exp(-(xs**2 + ys**2) / (2*sigma**2))
    gauss /= np.sum(gauss)
    flat = convolve(np.ones_like(image), gauss, mode="same")
    return convolve(image, gauss, mode="same") / flat

def sky_to_ra_dec(x, y, xcol, ycol):
    xs = (x - xcol.coord_ref_point) * xcol.coord_inc * np.pi / 180
    ys = (y - ycol.coord_ref_point) * ycol.coord_inc * np.pi / 180
    phi = np.arctan2(-xs, ys)
    theta = np.arctan2(1, np.sqrt(xs**2 + ys**2))
    dec_0 = ycol.coord_ref_value * np.pi / 180
    ra = xcol.coord_ref_value + 180 / np.pi * np.arctan2(
        -np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.cos(dec_0) - np.cos(theta) * np.sin(dec_0) * np.cos(phi)
    )
    dec = 180 / np.pi * np.arcsin(
        np.sin(theta) * np.sin(dec_0)
        + np.cos(theta) * np.cos(dec_0) * np.cos(phi)
    )
    return ra, dec

class ARF:
    def __init__(self, filename):
        with fits.open(filename) as hdul:
            energy = (hdul[1].data["ENERG_LO"] + hdul[1].data["ENERG_HI"]) / 2
            specresp = hdul[1].data["SPECRESP"]
        self.interp = interp1d(energy, specresp, fill_value=0, bounds_error=False)
    
    def __call__(self, energy):
        return self.interp(energy)
    
class Expmap:
    def __init__(self, filename):
        with fits.open(filename) as hdul:
            image = hdul[0].data
            wcs = WCS(hdul[0].header)
            upper_left, lower_right = wcs.all_pix2world([(0, 0), (image.shape[1]-1, image.shape[0]-1)], 0)
            ras = np.linspace(upper_left[0], lower_right[0], image.shape[0])
            decs = np.linspace(upper_left[1], lower_right[1], image.shape[1])
            self.expmap = RegularGridInterpolator((ras, decs), np.transpose(image), bounds_error=False, fill_value=0)

    def __call__(self, pos):
        """
        Return the exposure map at this position, in ra dec.
        """
        return self.expmap(pos)
    
def make_merged_image(args):
    # Load the information
    ixpe_arf = ARF(args.ixpe_arf)
    if args.expmap is not None:
        expmap = Expmap(args.expmap)
    else:
        expmap = None
    evt_ras = []
    evt_decs = []
    evt_weights = []
    for i in range(len(args.cxo_evt)):
        with fits.open(args.cxo_evt[i]) as hdul:
            these_ras, these_decs = sky_to_ra_dec(hdul[1].data["X"], hdul[1].data["Y"], hdul[1].columns["X"], hdul[1].columns["Y"])
            these_energies = np.array(hdul[1].data["ENERGY"]).astype(float) / 1000
            mask = (these_energies > args.elow) & (these_energies < args.ehigh)
        this_arf = ARF(args.cxo_arf[i])
        these_weights = ixpe_arf(these_energies) / this_arf(these_energies)
        if expmap is not None:
            these_weights /= expmap((these_ras, these_decs))

        evt_ras = np.concatenate([evt_ras, these_ras[mask]])
        evt_decs = np.concatenate([evt_decs, these_decs[mask]])
        evt_weights = np.concatenate([evt_weights, these_weights[mask]])

    # Convert to IXPE xy
    with fits.open(args.ixpe_evt) as hdul:
        xcol = hdul[1].columns["X"]
        ycol = hdul[1].columns["Y"]
        stretch = np.cos(ycol.coord_ref_value * np.pi / 180)
        ra_zero = (-xcol.coord_ref_point)/stretch * xcol.coord_inc + xcol.coord_ref_value
        dec_zero = (-ycol.coord_ref_point) * ycol.coord_inc + ycol.coord_ref_value
        ixpe_xs = -(evt_ras - ra_zero) * stretch * 3600 / IXPE_PIXEL_SIZE
        ixpe_ys = (evt_decs - dec_zero) * 3600 / IXPE_PIXEL_SIZE

    # Background subtract
    bkg_sb = 0 # Background flux per square arcsec
    if args.reg_bkg is not None:
        reg_bkg = leakagelib.Region.load(args.reg_bkg)
        mask = reg_bkg.check_inside_absolute(ixpe_xs, ixpe_ys)
        bkg_sb = np.sum(evt_weights[mask]) / (reg_bkg.area() * IXPE_PIXEL_SIZE**2)

    # Create the image
    mask = np.ones(len(evt_weights), bool)
    if args.reg_src is not None:
        reg_src = leakagelib.Region.load(args.reg_src)
        mask &= reg_src.check_inside_absolute(ixpe_xs, ixpe_ys)

    if args.width is None:
        delta = max(np.max(ixpe_xs) - np.min(ixpe_xs), np.max(ixpe_ys) - np.min(ixpe_ys)) * IXPE_PIXEL_SIZE
    else:
        delta = float(args.width)
    line = np.arange(0, delta/2, PIXEL_WIDTH) + PIXEL_WIDTH / 2
    line = np.concatenate([-np.flip(line), line])
    linex = line + (np.max(ixpe_xs) + np.min(ixpe_xs)) / 2 * IXPE_PIXEL_SIZE
    liney = line + (np.max(ixpe_ys) + np.min(ixpe_ys)) / 2 * IXPE_PIXEL_SIZE

    image = np.histogram2d(ixpe_xs[mask] * IXPE_PIXEL_SIZE, ixpe_ys[mask] * IXPE_PIXEL_SIZE, (linex, liney), weights=evt_weights[mask])[0]
    image -= bkg_sb * PIXEL_WIDTH * PIXEL_WIDTH
    image = blur(image, 1)
    image = np.maximum(image, 0)
    image = np.transpose(image)

    # Save the image
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [
        ra_zero - linex[len(linex)//2] / 3600 / stretch,
        dec_zero + liney[len(liney)//2] / 3600
    ]
    w.wcs.crpix = [len(linex)//2+0.5, len(liney)//2+0.5]
    w.wcs.cdelt = [-PIXEL_WIDTH / 3600, PIXEL_WIDTH / 3600]
    w.wcs.cunit = ["deg", "deg"]
    header = w.to_header()
    del header["LATPOLE"]
    del header["LONPOLE"]
    fits.writeto(args.output, image, header, overwrite=args.clobber)

def cxo_source(filename, data, offset=None):
    """
    Loads an image generated by leakagelib_cxo for use in LeakageLib fitting. If you applied any offsets to the data set, these offsets are applied to the image (that's why data is an argument). You can apply additional offsets too (units: IXPE pixels).

    Parameters
    ----------
    filename : str
        File name of the CXO image. This should have been created with leakagelib_cxo
    data : :class:`leakagelib.IXPEData`
        IXPE data set to create a source for
    offset : (float, float)
        Additional offset in units of arcseconds

    Returns
    -------
        :class:`leakagelib.Source`
    A Source object with the correct pixel scale for images generated by leakagelib_cxo

    """
    with fits.open(data.filename) as hdul:
        xcol = hdul[1].columns["X"]
        ycol = hdul[1].columns["Y"]
        stretch = np.cos(ycol.coord_ref_value * np.pi / 180)

    with fits.open(filename) as hdul:
        image = hdul[0].data
        wcs = WCS(hdul[0].header)
        upper_left, lower_right = wcs.all_pix2world([(0, 0), (image.shape[1]-1, image.shape[0]-1)], 0)
        old_ras = np.linspace(upper_left[0], lower_right[0], image.shape[0])
        old_decs = np.linspace(upper_left[1], lower_right[1], image.shape[1])
        old_x = xcol.coord_ref_point + (old_ras - xcol.coord_ref_value) / xcol.coord_inc * stretch
        old_y = ycol.coord_ref_point + (old_decs - ycol.coord_ref_value) / ycol.coord_inc
        old_x *= 2.6 # Convert to arcsec
        old_y *= 2.6

    # TODO there's a total offset. Compare the new / old images.

    # Move the image
    old_x += data.offsets[0]
    old_y += data.offsets[1]
    if offset is not None:
        old_y += offset[1]
        old_x += offset[0]
    old_interpolator = RegularGridInterpolator((old_x, old_y), np.transpose(image), fill_value=0, bounds_error=False)

    new_line = np.arange(1, 1+image.shape[0]//2).astype(float) * PIXEL_WIDTH
    new_line = np.concatenate([-np.flip(new_line), [0], new_line])
    xs, ys = np.meshgrid(new_line, new_line)
    image = old_interpolator((xs.reshape(-1), ys.reshape(-1))).reshape(image.shape)

    # Create the Source object
    return leakagelib.Source(image, data.use_nn, len(image), PIXEL_WIDTH)
    