"""
Fit for the polarization within a background region. This code is based heavily on the point source fit example. The reader should study that code first.
"""
import numpy as np
import sys
sys.path.append("../../..")
import leakagelib

def fit(energy_cut=(2,8)):
    # Load the data as in the point source example
    source = leakagelib.source.Source.no_image(False)
    datas = [leakagelib.IXPEData(source, (
        "data/ps/event_l2/ixpeps_det1_evt2_v00.fits",
        "data/ps/hk/ixpeps_det1_att_v00.fits",
    ), energy_cut=energy_cut, bin=False)]

    for data in datas:
        # Center coordinates on the object and cut to data within 280 arcsec
        data.iterative_centroid_center()
        data.retain(np.sqrt(data.evt_xs**2 + data.evt_ys**2) < 280)

    for data in datas:
        # NEW: Remove data within 60 arcsec of the center, cutting out the source.
        data.retain(np.sqrt(data.evt_xs**2 + data.evt_ys**2) > 60)

    # Make a "fit settings" object, which encodes fit components
    settings = leakagelib.ps_fit.FitSettings(datas)

    # Add a photon background component
    settings.add_background("bkg")
    settings.set_initial_qu("bkg", (0, 0)) # Allow the background polarization to be fitted, but set the initial guess to unpolarized.
    settings.fix_flux("bkg", 1) # As always, fix at least one component's flux

    settings.add_particle_source()

    # Our ROI is annular. There is no builtin function for an annular ROI, so we must tell the fitter the ROI mask.
    pixel_centers = settings.sources[0].pixel_centers
    # The apply_roi function expects an n x n image, where the coordinates of the ith row or column is pixel_centers[i]
    xs, ys = np.meshgrid(pixel_centers, pixel_centers)
    radii = np.sqrt(xs**2 + ys**2)
    roi_mask = (radii > 60) & (radii < 280)
    settings.apply_roi(roi_mask)

    # Set the spectrum model
    from ixpeobssim.irf import load_arf
    arf = load_arf()
    settings.set_spectrum("bkg", lambda e: arf(e) * e**-2.5)

    # Set up the fitter. Do not modify the settings object after this is done
    fitter = leakagelib.ps_fit.Fitter(datas, settings)
    print(fitter)
    fitter.display_sources("figs/sources.png") # Plot the fit models you're using

    result = fitter.fit()
    print(result)

if __name__ == "__main__":
    fit()