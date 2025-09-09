"""
Fit for the polarization of an extended source. See `point-source-fit.py` if you'd like the simplest introduction
"""
import numpy as np
import sys
sys.path.append("../../..")
import leakagelib

def fit(energy_cut=(2,8)):
    source = leakagelib.source.Source.no_image(False)

    # Load the extended source data
    datas = [leakagelib.IXPEData(source, (
        "data/extended/event_l2/ixpeextended_det1_evt2_v00.fits",
        "data/extended/hk/ixpeextended_det1_att_v00.fits",
    ), energy_cut=energy_cut, bin=False)]

    for data in datas:
        # This time, explicitly center since centering on the centroid of an extended source is prone to error
        data.explicit_center(300,300)

        # Cut to a 280 arcsecond circular aperture.
        data.retain(np.sqrt(data.evt_xs**2 + data.evt_ys**2) < 280)


    # Create an source component with a centered disk of radius 1 arcmin set to 1, and everything else set to zero
    pixel_size = 2.9729 # Use the pixel size of the sky PSF for best results
    source_pixel_edges = np.arange(0, 280+pixel_size, pixel_size)
    source_pixel_edges = np.concatenate([-np.flip(source_pixel_edges)[:-1], source_pixel_edges])
    source_pixel_centers = (source_pixel_edges[1:] + source_pixel_edges[:-1]) / 2
    xs, ys = np.meshgrid(source_pixel_centers, source_pixel_centers)
    image = (np.sqrt(xs**2 + ys**2) < 60).astype(float)

    # Create a source object for this image
    source = leakagelib.Source(image, False, len(image), pixel_size)

    # Add the source object to the fitter information
    settings = leakagelib.ps_fit.FitSettings(datas)
    settings.add_source(source, "src-ext")
    settings.fix_flux("src-ext", 1)

    # Finish setting up the fit settings as for the point source example
    settings.add_background("bkg")
    settings.fix_qu("bkg", (0, 0))
    settings.set_initial_flux("bkg", 1)
    
    settings.apply_circular_roi(280)

    from ixpeobssim.irf import load_arf
    arf = load_arf()
    settings.set_spectrum("bkg", lambda e: arf(e) * e**-2.5)
    settings.set_spectrum("src-ext", lambda e: arf(e) * e**-1.5)

    # Perform the fit. The true polarization of this simulated data set is Q=0.5, U=0
    fitter = leakagelib.ps_fit.Fitter(datas, settings)
    result = fitter.fit()
    print(result)

if __name__ == "__main__":
    fit()