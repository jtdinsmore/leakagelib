import numpy as np
from regions import Regions, PixCoord
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

IXPE_PIXEL_SIZE = 2.6

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
        if assert_format is not None and assert_format != self.fmt:
            raise Exception(f"The region {filename} was supposed to be in {assert_format} but it was actually in {self.fmt} format. Please change it.")
        text = self.fmt + ";\n" + text

        # Load the region
        regions = Regions.parse(text, format="ds9")
        self.regions = regions
        self.region = regions[0]
        for reg in regions:
            self.region &= reg

        # Get a tan projection to use sky coordinates
        if self.fmt == "fk5":
            # Get the center of the region
            try:
                center = reg.center
            except Exception:
                # fallback: average vertices if no analytic center
                for r in regions:
                    if hasattr(r, "vertices"):
                        center = r.vertices[0]
                        break
            pixel_size = 1/3600
            radius = 16/60
            npix = int(2*radius / pixel_size)
            if npix % 2 == 0:
                npix += 1

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
        if self.fmt == "image":
            reg = self.region
            pixel_center = 300
            pixel_area = (IXPE_PIXEL_SIZE / 3600)**2 # Square degrees
        else:
            reg = self.region.to_pixel(self.wcs)
            pixel_center = self.wcs.wcs.crpix[0]
            pixel_area = self.wcs.wcs.cdelt[1]**2 # Square degrees
        line = np.arange(2*pixel_center)
        xs, ys = np.meshgrid(line, line)
        coords = PixCoord(x=xs, y=ys)
        image = reg.contains(coords)
        return np.sum(image) * pixel_area