import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import convolve
from scipy.interpolate import interp1d, RegularGridInterpolator
from leakagelib.ixpe_data import IXPE_PIXEL_SIZE
import leakagelib

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
            ras = np.linspace(upper_left[0], lower_right[0], image.shape[1])
            decs = np.linspace(upper_left[1], lower_right[1], image.shape[0])
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
        cxo_expmap = Expmap(args.expmap)
    else:
        cxo_expmap = None
    
    # Get events
    evt_ras = []

    evt_decs = []
    evt_weights = []
    for i in range(len(args.cxo_evt)):
        with fits.open(args.cxo_evt[i]) as hdul:
            these_ras, these_decs = sky_to_ra_dec(hdul[1].data["X"], hdul[1].data["Y"], hdul[1].columns["X"], hdul[1].columns["Y"])
            these_energies = np.array(hdul[1].data["ENERGY"]).astype(float) / 1000
            mask = (these_energies > float(args.elow)) & (these_energies < float(args.ehigh))

        this_arf = ARF(args.cxo_arf[i])
        these_weights = ixpe_arf(these_energies[mask]) / this_arf(these_energies[mask])
        if cxo_expmap is not None:
            these_weights /= cxo_expmap((these_ras[mask], these_decs[mask]))
        these_weights[~np.isfinite(these_weights)] = 0

        evt_ras = np.concatenate([evt_ras, these_ras[mask]])
        evt_decs = np.concatenate([evt_decs, these_decs[mask]])
        evt_weights = np.concatenate([evt_weights, these_weights])

    # Background subtract
    bkg_sb = 0 # Background flux per square arcsec
    if args.reg_bkg is not None:
        reg_bkg = leakagelib.Region(args.reg_bkg, assert_format="fk5")
        mask = reg_bkg.contains(evt_ras, evt_decs)
        bkg_sb = np.sum(evt_weights[mask]) / (reg_bkg.area() * 3600**2)

    # Get the events within the image
    mask = np.ones(len(evt_weights), bool)
    if args.reg_src is not None:
        reg_src = leakagelib.Region(args.reg_src, assert_format="fk5")
        mask &= reg_src.contains(evt_ras, evt_decs)

    # Convert to IXPE xy, centered on the image center
    with fits.open(args.ixpe_evt) as hdul:
        xcol = hdul[1].columns["X"]
        ycol = hdul[1].columns["Y"]
        stretch = np.cos(ycol.coord_ref_value * np.pi / 180)
        ra_zero = (-xcol.coord_ref_point)/stretch * xcol.coord_inc + xcol.coord_ref_value
        dec_zero = (-ycol.coord_ref_point) * ycol.coord_inc + ycol.coord_ref_value
        ixpe_xs = -(evt_ras - ra_zero) * stretch * 3600 / IXPE_PIXEL_SIZE
        ixpe_ys = (evt_decs - dec_zero) * 3600 / IXPE_PIXEL_SIZE
    ixpe_xs = (ixpe_xs - float(args.centerx))[mask]
    ixpe_ys = (ixpe_ys - float(args.centery))[mask]

    # Create the image
    if args.width is None:
        halfwidth = max(np.max(np.abs(ixpe_xs)), np.max(np.abs(ixpe_ys)))
    else:
        halfwidth = float(args.width) / IXPE_PIXEL_SIZE/ 2
    pixel_edges = np.arange(0, halfwidth, PIXEL_WIDTH/IXPE_PIXEL_SIZE) + PIXEL_WIDTH/IXPE_PIXEL_SIZE / 2
    pixel_edges = np.concatenate([-np.flip(pixel_edges), pixel_edges])
    image = np.histogram2d(ixpe_xs, ixpe_ys, (pixel_edges, pixel_edges), weights=evt_weights[mask])[0]
    image -= bkg_sb * PIXEL_WIDTH**2
    image = blur(image, 1)
    image = np.maximum(image, 0)
    image = np.transpose(image)

    # Save the image
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [
        ra_zero - args.centerx * IXPE_PIXEL_SIZE / 3600 / stretch,
        dec_zero + args.centerx * IXPE_PIXEL_SIZE / 3600
    ]
    w.wcs.crpix = [len(pixel_edges)//2+0.5, len(pixel_edges)//2+0.5]
    w.wcs.cdelt = [-PIXEL_WIDTH / 3600, PIXEL_WIDTH / 3600]
    w.wcs.cunit = ["deg", "deg"]
    header = w.to_header()
    del header["LATPOLE"]
    del header["LONPOLE"]
    try:
        fits.writeto(args.output, image, header, overwrite=args.clobber)
    except OSError:
        raise Exception(f"Could not write to {args.output}. Please pass clobber as a command line argument to overwrite files.")