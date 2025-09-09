import numpy as np
from astropy.io import fits

def get_times(f):
    """Load the times in the file f as an array"""
    with fits.open(f) as hdul:
        times = hdul[1].data["TIME"]
    return np.array(times)


def write_probs(infile, outfile, bg_probs):
    '''Write probabilities to outfile
    # Arguments
        * infile: file name which will be masked
        * output file: file which will be written to. If the file already exists, it will be overwritten.
        * mask: boolean array created by `get_mask`.
    '''
    
    with fits.open(infile) as hdul:
        if len(hdul[1].data) != len(bg_probs):
            raise Exception("The L1 file you provided does not have the same number of events as the file you are masking..")

        # Copy all the file HDUs
        hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])

        bg_prob_col = fits.Column(name='BG_PROB', format='E', array=bg_probs)
        new_cols = hdul[1].columns + bg_prob_col
        hdul_copy[1] = fits.BinTableHDU.from_columns(new_cols, header=hdul[1].header)

        # Save to a new file
        hdul_copy.writeto(outfile, overwrite=True)