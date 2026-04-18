import numpy as np
from regions import Regions, PixCoord
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

class Region:
    """
    Class to store CIAO-format regions. Either fk5 or physical coordinates are allowed.

    Parameters
    ----------
    filename : str
        Name of the region file
    assert_format : str, optional
        Either "fk5" or "image". If this variable is set, any region which is not of format `assert_format` will throw an error. Default: None.
    """
    def __init__(self, filename, assert_format=None):
        with open(filename) as f:
            text = f.read()

        # Guess the format
        if ":" in text:
            self.fmt = "fk5"
        else:
            self.fmt = "image"

            raise Exception(f"The region {filename} was supposed to be in {assert_format} but it was actually in {self.fmt} format. Please change it.")
        text = self.fmt + ";\n" + text

        # Load the region
        regions = Regions.parse(text, format="ds9")
        self.region = regions[0]
        for reg in regions:
            self.region |= reg

        # Get a tan projection to use sky coordinates
        if self.fmt == "fk5":
            # Get the center of the region
            try:
                center = reg.center
            except Exception:
                # fallback: average vertices if no analytic center
                coords = []
                for r in regions:
                    if hasattr(r, "vertices"):
                        coords.append(r.vertices)
                center = np.mean(coords, axis=0)
                print(center.shape, "2")# TODO

            pixel_size = 1/3600
            radius = 10/60
            npix = int(2*radius / pixel_size)

            self.wcs = WCS(naxis=2)
            self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            self.wcs.wcs.crval = [center.ra.deg, center.dec.deg]
            self.wcs.wcs.crpix = [npix/2, npix/2]
            self.wcs.wcs.cdelt = [-pixel_size, pixel_size]

    def contains(self, x, y):
        """
        Check if the coordinate x, y is contained in the region. If the region is physical-format, x and y should be in units of pixels. If it's in fk5 format, the units should be degrees.
        """
        if self.fmt == "image":
            coord = PixCoord(x=x, y=y)
            return self.region.contains(coord)
        else:
            coord = SkyCoord(ra=x, dec=y, unit="deg", frame="fk5")
            return self.region.contains(coord, self.wcs)
    
    def area(self):
        return self.region.area()