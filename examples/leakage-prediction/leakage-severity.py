'''
Make a plot showing the severity of leakage for an unpolarized point source.
Specifically, plot a histogram of the leaked PD (divided by mu)
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../..")
import leakagelib

leakagelib.funcs.override_matplotlib_defaults()

DETECTOR = 3
SOURCE_SIZE = 101 # Pixels
PIXEL_SIZE = 1 # arcsec. Since we aren't using data, it's OK to make it this small

def get_pd_image(use_nn, spectrum):
    # Source distribution. Delta indicates a delta function, or point source. 
    # If you wish to use an extended source, replace the Source object with another class.
    source = leakagelib.Source.delta(
        use_nn, # Whether to use neural net leakage parameters or not
        SOURCE_SIZE, # Number of pixels across the image
        PIXEL_SIZE # Width of each pixel in arcsec
    )
    # # Example non-point source. Gives weaker leakage patterns
    # source = leakagelib.Source.gaussian(
    #     use_nn, SOURCE_SIZE, PIXEL_SIZE,
    #     10, # Gaussian width
    # )

    # Load the PSF (you should not change this; other PSF models are less accurate.)
    psf = leakagelib.PSF.sky_cal(
        DETECTOR, # Detector index
        source, # Source class
        0 # Observation orientation (set to zero, since we don't have to compare to data)
    )

    # Compute the leakage predictions
    pred_i, q_norm, u_norm = source.compute_leakage(
        psf,
        spectrum,
        normalize=True # Normalize the q and u output
    )

    # Divide by mu so that results are comparable to physical polarizations
    q_norm, u_norm = source.divide_by_mu(q_norm, u_norm, spectrum)

    pd = np.sqrt(q_norm**2 + u_norm**2)

    return pd

def plot(settings):
    '''
    Make the plot. Settings is an array containing tuples of (use_nn, spectrum, histogram color, histogram label).
    '''

    pd_bins = np.linspace(0, 75, 100)

    fig, ax = plt.subplots()
    for (use_nn, spectrum, color, label) in settings:
        image = get_pd_image(use_nn, spectrum)
        output = []
        line_drawn = False
        for pd in pd_bins:
            polarized_fraction = np.mean(image * 100 > pd)
            output.append(polarized_fraction)
            if polarized_fraction < 0.05 and not line_drawn:
                ax.axvline(pd, color=color, lw=1)
                line_drawn = True
                ax.text(pd+0.5, 0.95, f"95% {label}", ha='left', va='top', rotation=90, size=24, color=color)
        ax.plot(pd_bins, output, lw=3, color=color, label=label)

    ax.set_ylim(0, 1)
    ax.set_xlim(pd_bins[0], pd_bins[-1])
    ax.set_xlabel("$\\mu$-corrected polarization degree (PD) [%]")
    ax.set_ylabel("Image fraction with larger PD")
    ax.legend()

    fig.savefig("figs/severity.png")
    # fig.savefig("figs/severity.pdf")

if __name__ == "__main__":
    settings = [
        (False, leakagelib.Spectrum.from_power_law_index(2), "#5F3694", "Mom"),
        (True, leakagelib.Spectrum.from_power_law_index(2), "#CA3142", "NN")
    ]

    plot(settings)