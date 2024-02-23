import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
sys.path.append("../..")
import leakagelib
import time

SOURCE_SIZE = 53 # pixels
PIXEL_SIZE = 2.8 # arcsec
VMAX = 0.5

SPECTRUM = leakagelib.Spectrum.from_power_law_index(2)
leakagelib.funcs.override_matplotlib_defaults() # Set the matplotlib defaults to Jack's settings

if __name__ == "__main__":
    # Load the nebula files
    source = leakagelib.Source.load_file(
        "data/pwn-i.fits",
        False,               # Predict leakage assuming Moments data
        SOURCE_SIZE,  # Number of spatial bins to put in a single row of your image. The image is assumed to be square
        PIXEL_SIZE # The size of each pixel in arcsec. Together this and the previous argument multiply to give the width of the image in arcsec
    )
    source.polarize_file("data/pwn-qu.fits")
    
    # Predict leakage for each source
    fig, axs = plt.subplots(ncols=3,nrows=4, figsize=(12, 17), sharex=True, sharey=True, gridspec_kw=dict(height_ratios=(1,1,1,1.4)))

    axs[0,0].pcolormesh(source.pixel_centers, source.pixel_centers, np.log10(source.source))
    axs[0,1].pcolormesh(source.pixel_centers, source.pixel_centers, source.q_map, vmin=-VMAX,vmax=VMAX, cmap="RdBu")
    axs[0,2].pcolormesh(source.pixel_centers, source.pixel_centers, source.u_map, vmin=-VMAX,vmax=VMAX, cmap="RdBu")

    for det in range(3):
        # Load the PSF
        psf = leakagelib.PSF.sky_cal(
            det,                # Use the given detector index
            source,             # Use the Source object just created
            det * np.pi / 3 * 2 # Rotate the source by this amount
        )

        start = time.time()
        # Get the predicted detection maps for q and u
        i, q_norm, u_norm = source.compute_leakage(
            psf,                # Use the PSF that was just loaded
            SPECTRUM,           # Use an example power-law spectrum
            normalize=True     # Normalize the output q and u. Off by default
        )
        # print(f"Took {(time.time() - start) / SOURCE_SIZE**2 * 1000} s per 1000 pixels")

        # OPTIONAL: Divide these maps by the detector modulation factor. After this division, the
        # PD = sqrt(q_norm**2 + u_norm**2) is equal to the point source polarization for an
        # aperture large enough that leakage effects can be neglected. Before the division, the maps
        # predict the actual detected polarizations and are therefore lowered by the modulation
        # factor mu.
        #
        # This division step is likely necessary if comparing with other tools. For comparison with
        # unweighted IXPE data, it should not be done.
        q_norm, u_norm = source.divide_by_mu(q_norm, u_norm, SPECTRUM)

        ci = axs[det+1,0].pcolormesh(source.pixel_centers, source.pixel_centers, np.log10(i))
        axs[det+1,1].pcolormesh(source.pixel_centers, source.pixel_centers, q_norm, vmax=VMAX, vmin=-VMAX, cmap="RdBu")
        cqu = axs[det+1,2].pcolormesh(source.pixel_centers, source.pixel_centers, u_norm, vmax=VMAX, vmin=-VMAX, cmap="RdBu")

    for ax in axs.reshape(-1):
        ax.set_aspect("equal")
        ax.set_xlim(source.pixel_centers[-1], source.pixel_centers[0])
        ax.set_ylim(source.pixel_centers[0], source.pixel_centers[-1])

    axs[0,0].set_ylabel("Truth")
    axs[1,0].set_ylabel("Detector 1")
    axs[2,0].set_ylabel("Detector 2")
    axs[3,0].set_ylabel("Detector 3")
    axs[0,0].set_title("Log I")
    axs[0,1].set_title("q (normalized)")
    axs[0,2].set_title("u (normalized)")
    axs[-1,0].set_xlabel("[arcsec]")

    fig.colorbar(cqu, ax=axs[-1,(1,2)], orientation="horizontal", aspect=40)
    cbari = fig.colorbar(ci, ax=axs[-1,0], orientation="horizontal")
    cbari.set_ticks([])

    if not os.path.exists("figs"):
        os.mkdir("figs")

    fig.savefig("figs/predict.png")
    # fig.savefig("figs/predict.pdf")
    