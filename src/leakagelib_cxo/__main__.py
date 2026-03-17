import argparse, logging, os
from .cxo_utils import make_merged_image

logger = logging.getLogger("leakagelib")

def check_args(args):
    if len(args.cxo_evt) != len(args.cxo_arf):
        raise Exception("You must pass the same number of ARFs as event files. One ARF for each event file")
    for i in range(len(args.cxo_evt)):
        if not os.path.exists(args.cxo_evt[i]):
            raise Exception(f"{args.cxo_evt[i]} does not exist")
        if not os.path.exists(args.cxo_arf[i]):
            raise Exception(f"{args.cxo_arf[i]} does not exist")

    if args.expmap is not None:
        if not os.path.exists(args.expmap):
            raise Exception(f"{args.expmap} does not exist")
    else:
        logger.warning("Exposure map corrections will not be made because you did not pass an exposure map")
        
    if not os.path.exists(args.ixpe_arf):
        raise Exception(f"{args.ixpe_arf} does not exist")
    if not os.path.exists(args.ixpe_evt):
        raise Exception(f"{args.ixpe_evt} does not exist")

    if args.elow >= args.ehigh:
        raise Exception("Your low energy must be smaller than your high energy")
    if  float(args.ehigh) > 10 or float(args.elow) < 1:
        logger.warning("Your energies should be in keV. Please check to make sure they are correct.")

    if args.reg_src is not None:
        if not os.path.exists(args.reg_src):
            raise Exception(f"{args.reg_src} does not exist")
    else:
        logger.warning("You did not provide a source region. The Chandra image will display the entire field")
    if args.reg_bkg is not None:
        if not os.path.exists(args.reg_src):
            raise Exception(f"{args.reg_src} does not exist")
    else:
        logger.warning("You did not provide a background region. The Chandra image will not be background subtracted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog='clean',
                description='Creates a CXO image, adjusted to the IXPE band. Before running this command, you should use the ciao tools merge_obs and mkwarf to create a merged image (including an exposure map) and an arf for each observaton. These are required as inputs for this script.')
    parser.add_argument('--cxo-evt', nargs='+', required=True, help="List of Chandra event files to use")
    parser.add_argument('--cxo-arf', nargs='+', required=True, help="List of Chandra ARFs to use")
    parser.add_argument('--ixpe-evt', required=True, help="IXPE event file (only one is necessary, for simplicity. Try DU1)")
    parser.add_argument('--ixpe-arf', required=True, help="IXPE ARF (only one is necessary, for simplicity. Try DU1)")
    parser.add_argument('--expmap', help="Chandra merged observation ARF (Optional)")
    parser.add_argument("--output", required=True, help="Name of the output file")
    parser.add_argument("--width", help="Width of the image, in arcseconds. Default: as big as the CXO image (Optional)")
    parser.add_argument("--elow", default=2, help="Low end of the energy range (keV). Default: 2")
    parser.add_argument("--ehigh", default=8, help="High end of the energy range (keV). Default: 8")
    parser.add_argument("--reg-src", help="Source region. Should be saved in CIAO format with IXPE physical coordinates (Optional)")
    parser.add_argument("--reg-bkg", help="Background region. Should be saved in CIAO format with IXPE physical coordinates (Optional)")
    parser.add_argument("--clobber", action=argparse.BooleanOptionalAction, help="Overwrite files")

    args = parser.parse_args()

    check_args(args)
    make_merged_image(args)

    print(f"""Image saved to {args.output}. To use it, load the source with LeakageLib using the following code:
          
from leakagelib_cxo import cxo_source
source = cxo_source("{args.output}", use_nn)

This gives a leakagelib.Source object, which you can use as a source in your LeakageLib fit.""")