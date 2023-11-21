NN_SIGMA_PERP = 1.08023476
NN_SIGMA_PERP_SLOPE = 0.03332729
NN_KURTOSIS_RATIO = 1.31607401295 # Theoretical prediction

MOM_SIGMA_PERP =  0.55811705
MOM_SIGMA_PERP_SLOPE = 0.27555888
MOM_KURTOSIS_RATIO = 1.25258597

DATA_DIRECTORIES = [
    "/Users/jtd/Documents/research/ixpepl/data",
    "/Users/jtd/Documents/research/ixpepl/data/l2-only",
]
LEAKAGE_DATA_DIRECTORY = '/'.join(__file__.split('/')[:-2]) + "/data"
GROUND_PSF_DIRECTORY = "../../data/ground-psfs"