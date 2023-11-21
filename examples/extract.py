import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
sys.path.append("../..")
import leakagelib

SOURCE_SIZE = 55 # pixels # 41
IMAGE_SIZE = 2.8 # arcsec # 4
PIXEL_SIZE = 2.8 # arcsec # 4
VMAX = 0.5

SPECTRUM = leakagelib.Spectrum.from_power_law_index(2)
leakagelib.funcs.override_matplotlib_defaults() # Set the matplotlib defaults to Jack's settings

if __name__ == "__main__":
    # Load the nebula files
    source = leakagelib.Source.load_file(
        "data/pwn-i.fits",
        False,  # True if you would like to use neural net-reconstructed data in the future, and false if you would like to use Mom-reconstructed data.
        SOURCE_SIZE,  # Number of spatial bins to put in a single row of your image. The image is assumed to be square
        PIXEL_SIZE # The size of each pixel in arcsec. Together this and the previous argument multiply to give the width of the image in arcsec
    )
    source.polarize_file("data/pwn-qu.fits")
    mock_is = np.load("data/mock_is.npy")
    mock_qs = np.load("data/mock_qs.npy")
    mock_us = np.load("data/mock_us.npy")

    # Load the PSFs
    psfs = []
    for det in range(3):
        psfs.append(leakagelib.PSF.sky_cal(det, source, det * np.pi / 3 * 2))

    # Manually set the inertia of the point source
    # inertia = np.ones((2, mock_qs[0].shape[0], mock_qs[0].shape[1]))
    # inertia[:,source.source > (np.max(source.source) / 200)] = 0.1
    # inertia[:,source.source > (np.max(source.source) / 100)] = 0.2
    # inertia[:,source.source > (np.max(source.source) / 50)] = 0.5
    # inertia[:,source.source > (np.max(source.source) / 2)] = 0.01

    # Perform the fit
    extracted_q, extracted_u, anim = leakagelib.fit_extended(
        source, psfs, SPECTRUM,                                     # Properties of the source
        mock_is, mock_qs, mock_us,                                  # Leakage-containing observations
        initial_source_pol=None,                                    # Use the default starting point for the fitter
        # inertia=inertia,                                            # Optional argument to prevent some pixels from being numerically unstable
        num_iter=5000, max_rate=2e-2, regularize_coeff=0.4,         # Fitter settings
        report_frequency=50,                                        # Saves a snapshot every frame. Set to None to improve speed
    )

    # Save the gif
    if not os.path.exists("figs"):
        os.mkdir("figs")
    if anim is not None:
        anim.save(f"figs/animation.gif")

    # Plot the result
    source.polarize_file("data/pwn-qu.fits") # Reset to the true polarization
    fig, axs = plt.subplots(figsize=(13, 12), ncols=3, nrows=3, sharex=True, sharey=True, gridspec_kw=dict(width_ratios=(1.33, 1, 1)))

    true_row = source.pixel_centers
    mock_row = true_row

    true_i = source.source
    true_q = source.q_map
    true_u = source.u_map
    

    c_i = axs[0,0].pcolormesh(true_row, true_row, true_i, norm=mpl.colors.LogNorm(), rasterized=True)
    axs[0,1].pcolormesh(mock_row, mock_row, mock_is[2], norm=mpl.colors.LogNorm(), rasterized=True)
    axs[0,2].axis(False)
    c_qu = axs[1,0].pcolormesh(true_row, true_row, true_q, vmax=VMAX, vmin=-VMAX, cmap="RdBu", rasterized=True)
    axs[1,1].pcolormesh(mock_row, mock_row, mock_qs[2], vmax=VMAX, vmin=-VMAX, cmap="RdBu", rasterized=True)
    axs[1,2].pcolormesh(mock_row, mock_row, extracted_q, vmax=VMAX, vmin=-VMAX, cmap="RdBu", rasterized=True)
    axs[2,0].pcolormesh(true_row, true_row, true_u, vmax=VMAX, vmin=-VMAX, cmap="RdBu", rasterized=True)
    axs[2,1].pcolormesh(mock_row, mock_row, mock_us[2], vmax=VMAX, vmin=-VMAX, cmap="RdBu", rasterized=True)
    axs[2,2].pcolormesh(mock_row, mock_row, extracted_u, vmax=VMAX, vmin=-VMAX, cmap="RdBu", rasterized=True)

    for ax in axs.reshape(-1):
        ax.set_aspect("equal")
        ax.set_xlim(mock_row[-1], mock_row[0])
        ax.set_ylim(mock_row[0], mock_row[-1])
    axs[2,0].set_xlabel("[arcsec]")

    axs[0,0].set_title("True")
    axs[0,1].set_title("Observed (DU 3)")
    axs[1,2].set_title("Extracted")

    axs[0,0].set_yticks([])
    axs[1,0].set_yticks([])
    axs[2,0].set_yticks([])

    fig.colorbar(c_i , ax=axs[0,0], location='left', label="$I$")
    fig.colorbar(c_qu, ax=axs[1,0], location='left', label="$q$ (normalized)")
    fig.colorbar(c_qu, ax=axs[2,0], location='left', label="$u$ (normalized)")

    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig.savefig(f"figs/extract.png")
    # fig.savefig(f"figs/extract.pdf")