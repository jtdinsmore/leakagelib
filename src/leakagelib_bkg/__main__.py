import argparse
from . import flag_background

if __name__ == "__main__":
    print("STARTING")
    parser = argparse.ArgumentParser(
                prog='clean',
                description='Flags background particles in IXPE datasets with a probability (BG_CHAR column) of being background. l1 should be the level 1 file, l2 should be the level 2 file you wish to add background probabilities to, and outfile is the output. If outfile already exists, it will be overwritten.')
    parser.add_argument("l2", help="Name of the input l2 file")
    parser.add_argument("l1", help="Name of the input l1 file")
    parser.add_argument("outfile", help="Name of the output file")

    args = parser.parse_args()

    print(f"L1 file: {args.l1}")
    print(f"L2 file: {args.l2}")
    flag_background(args.l1, args.l2, args.outfile)
    print(f"Wrote background probabilities to {args.outfile}")