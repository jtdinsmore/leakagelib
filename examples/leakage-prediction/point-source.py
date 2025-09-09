import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("../../..")
import leakagelib

SOURCE_SIZE = 21 # pixels. For a delta function source, make sure this is odd.
PIXEL_SIZE = 3 # arcsec
DETECTOR = 2 # Second detector

leakagelib.funcs.override_matplotlib_defaults() # Set the matplotlib defaults to Jack's settings

def plot_images(source, ixpe_data, pred_i, pred_q, pred_u):
    # Plotting code for the sake of this example.
    fig, axs = plt.subplots(ncols=3,nrows=3, figsize=(14, 12), sharex=True, sharey=True, gridspec_kw=dict(width_ratios=(1.25,1,1.25)))

    vmax = np.max(np.abs([ixpe_data.q, ixpe_data.u, pred_q, pred_u]))
    vmax_delta_i = np.max(np.abs(ixpe_data.i - pred_i))
    vmaxi = np.max([ixpe_data.i, pred_i])

    ci = axs[0,0].pcolormesh(source.pixel_centers, source.pixel_centers, ixpe_data.i, vmin=0, vmax=vmaxi)
    axs[0,1].pcolormesh(source.pixel_centers, source.pixel_centers, ixpe_data.q, vmax=vmax, vmin=-vmax, cmap="RdBu")
    axs[0,2].pcolormesh(source.pixel_centers, source.pixel_centers, ixpe_data.u, vmax=vmax, vmin=-vmax, cmap="RdBu")

    axs[1,0].pcolormesh(source.pixel_centers, source.pixel_centers, pred_i, vmin=0, vmax=vmaxi)
    axs[1,1].pcolormesh(source.pixel_centers, source.pixel_centers, pred_q, vmax=vmax, vmin=-vmax, cmap="RdBu")
    axs[1,2].pcolormesh(source.pixel_centers, source.pixel_centers, pred_u, vmax=vmax, vmin=-vmax, cmap="RdBu")

    cr = axs[2,0].pcolormesh(source.pixel_centers, source.pixel_centers, ixpe_data.i - pred_i, cmap="BrBG", vmax=vmax_delta_i, vmin=-vmax_delta_i)
    axs[2,1].pcolormesh(source.pixel_centers, source.pixel_centers, ixpe_data.q - pred_q, vmax=vmax, vmin=-vmax, cmap="RdBu")
    cqu=axs[2,2].pcolormesh(source.pixel_centers, source.pixel_centers, ixpe_data.u - pred_u, vmax=vmax, vmin=-vmax, cmap="RdBu")

    for ax in axs.reshape(-1):
        ax.set_aspect("equal")
        ax.set_xlim(source.pixel_centers[-1], source.pixel_centers[0])
        ax.set_ylim(source.pixel_centers[0], source.pixel_centers[-1])

    axs[0,0].set_ylabel("Observed")
    axs[1,0].set_ylabel("Predicted")
    axs[2,0].set_ylabel("Residuals\n(Observed - Predicted)")
    axs[0,0].set_title("I")
    axs[0,1].set_title("Q (Unnormalized)")
    axs[0,2].set_title("U (Unnormalized)")
    axs[2,0].set_xlabel("[arcsec]")

    fig.colorbar(cqu, ax=axs[:,-1], aspect=60)
    fig.colorbar(cr, ax=axs[2,0])
    fig.colorbar(ci, ax=axs[(0,1), 0], aspect=40)

    if not os.path.exists("figs"):
        os.mkdir("figs")

    fig.savefig("figs/point-source.png")
    # fig.savefig("figs/point-source.pdf")
    

if __name__ == "__main__":
    # Get a point source (delta function) Source object
    source = leakagelib.Source.delta(
        False,  # True if you would like to use neural net-reconstructed data in the future, and False if you would like to use Mom-reconstructed data.
        SOURCE_SIZE,  # Number of spatial bins to put in a single row of your image. The image is assumed to be square
        PIXEL_SIZE # The size of each pixel in arcsec. Together this and the previous argument multiply to give the width of the image in arcsec
    )

    # Use this line of code to give a uniform polarization to the whole image
    # source.polarize_net((0.25, -0.25))

    # Load the data. This uses LeakageLib's in-built data finding system. To use it, edit the DATA_DIRECTORIES variable in settings.py to point to the folder(s) that contains your IXPE data. If you don't want to use this and instead want to point directly to your own files, use leakagelib.IXPEData.load_all_detectors_with_path.
    # Make sure you have downloaded the HK data.
    ixpe_datas = leakagelib.IXPEData.load_all_detectors(source, "01002401") # GX 9+9
    # ixpe_datas = leakagelib.IXPEData.load_all_detectors(source, "01002601") # GX 301-2
    # ixpe_datas = leakagelib.IXPEData.load_all_detectors(source, "02001901") # LMC X1
    # ixpe_datas = leakagelib.IXPEData.load_all_detectors(source, "02002399") # 4U 1820-303
    
    ixpe_data = ixpe_datas[DETECTOR-1] # It's possible to load data for only one detector, but I'm loading all three and discarding all but the detector we want

    ixpe_data.centroid_center()

    # Load the PSF
    psf = leakagelib.PSF.sky_cal(
        DETECTOR,                       # Use the given detector index
        source,                         # Use the Source object just created
        ixpe_datas[DETECTOR-1].rotation   # Rotate the source by this amount
    )

    # Compute predictions
    pred_i, pred_q, pred_u = source.compute_leakage(
        psf,                # Use the PSF that was just loaded
        ixpe_data.spectrum, # Use an example power-law spectrum
        normalize=False     # Compute the unnormalized coefficients
    )

    # Adjust the prediction and data to the same number of counts
    pred_q *= np.nansum(ixpe_data.i) / np.nansum(pred_i)
    pred_u *= np.nansum(ixpe_data.i) / np.nansum(pred_i)
    pred_i *= np.nansum(ixpe_data.i) / np.nansum(pred_i)

    plot_images(source, ixpe_data, pred_i, pred_q, pred_u)