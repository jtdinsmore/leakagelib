# This variable contains the directories in which IXPE data is stored. Specifically, if your data is stored in "DIR/OBS_ID/event_l2/FITS_FILE.fits", then DATA_DIRECTORIES should contain "DIR" only.
DATA_DIRECTORIES = [
    "/Users/jtd/Documents/research/ixpepl/data",
]

# Do not change
NN_SIGMA_PARALLEL_SCALE = 1.09686156
NN_KURT_SCALE = 0.91136976
MOM_SIGMA_PARALLEL_SCALE = 1.07266766
MOM_KURT_SCALE = 0.86951316

# Do not change
LEAKAGE_DATA_DIRECTORY = '/'.join(__file__.split('/')[:-2]) + "/data"
GROUND_PSF_DIRECTORY = "../../data/ground-psfs"