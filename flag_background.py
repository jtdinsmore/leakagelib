import numpy as np
import os, argparse
from astropy.io import fits
from src.bkg.load_tracks import load_tracks, associate_events
from src.bkg.nn import Model
from src.bkg.io import write_probs, get_times

CHUNK_SIZE = 2048

def flag_background(l1_file, l2_file, outfile):
    """Write the probability of each event being background to the BG_PROB column of `outfile`. The
    rest of the event data is taken from the provided l2_file. The tracks are taken from the
    l1_file.
    
    If the l1_file and l2_file have the same number of rows, it will be assumed that rows of the
    same index correspond to the same event. If they have different numbers of rows, events in the
    l1 and l2 files will be associated by their time. If there are multiple events with the same
    time, the routine will emit a warning.
    """
    if not os.path.exists(l1_file):
        raise Exception(f"Could not find the l1 file {l1_file}")
    if not os.path.exists(l2_file):
        raise Exception(f"Could not find the input file {l2_file}")
    
    model = Model() # Load the NN background flagging model
    l1_times = get_times(l1_file)
    l2_times = get_times(l2_file)

    l1_indices = associate_events(l1_times, l2_times)

    # Load the track images from the l1 file and run the NN
    bg_probs = []
    with fits.open(l1_file, memmap=True) as hdul:
        for chunk_index in range(len(l2_times) // CHUNK_SIZE + 1):
            start_index = chunk_index * CHUNK_SIZE
            end_index = min((chunk_index + 1) * CHUNK_SIZE, len(l2_times))
            chunk_tracks = load_tracks(hdul, l1_indices[start_index:end_index])
            bg_probs.append(model(chunk_tracks))
    bg_probs = np.concatenate(bg_probs)

    write_probs(l2_file, outfile, bg_probs) # Write the probs to the BG_PROB column of outfile.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog='clean',
                description='Flags background particles in IXPE datasets with a probability (BG_PROB column) of being background. l1 should be the level 1 file, l2 should be the level 2 file you wish to add background probabilities to, and outfile is the output. If outfile already exists, it will be overwritten.')
    parser.add_argument("l2", help="Name of the input l2 file")
    parser.add_argument("l1", help="Name of the input l1 file")
    parser.add_argument("outfile", help="Name of the output file")

    args = parser.parse_args()

    print(f"L1 file: {args.l1}")
    print(f"L2 file: {args.l2}")
    flag_background(args.l1, args.l2, args.outfile)
    print(f"Wrote background probabilities to {args.outfile}")