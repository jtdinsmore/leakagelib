import numpy as np
import os, sys, logging
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve
from astropy.io import fits
from astropy.wcs import WCS
from ixpeobssim.irf import load_vign
from .region import Region
from .settings import DATA_DIRECTORIES
from .modulation import get_mom_modf, get_nn_modf

IXPE_PIXEL_SIZE = 2.6 # arcsec

logger = logging.getLogger("leakagelib")

class IXPEData:
    """
    A class containing the data of a single pointing and a single detector.

    Attributes
    ----------
    det : int
        The detector number (1, 2, or 3).
    filename : str
        The FITS file representing this data set.
    obs_id : str
        The observation ID.

    Notes
    -----
    To construct an :class:`leakagelib.IXPEData`, you should use the :meth:`leakagelib.IXPEData.load_all_detectors` or :meth:`leakagelib.IXPEData.load_all_detectors_with_path` functions, or this constructor.

    Event fields
    ------------
    evt_xs, evt_ys, evt_qs, evt_us, evt_energies, evt_pis, evt_times, evt_ws, evt_mus, evt_bg_chars, evt_exposures : array-like
        Properties of the individual events. **Positions are measured in arcseconds**. q and u are the IXPE-standard 2cos(psi) and 2sin(psi). Energies are measured in keV. Times are measured in seconds since MJDREF. To retain events, use the :meth:`leakagelib.IXPEData.retain` or :meth:`leakagelib.IXPEData.retain_region` methods.
        

    offsets : tuple of float
        Location of the current origin in physical coordinates. Defaults to (0, 0).

    Notes
    -----
        **Do not manually edit** the evt_xs, evt_ys, evt_qs, evt_us, evt_pis, or evt_energies fields. `IXPEData` stores them to increase computational speed and editing them may cause the `IXPEData` object to go out of sync.
    """

    def load_all_detectors(obs_id, event_dir="event_l2"):
        '''
        Load all detectors corresponding to a specific observation ID.

        Parameters
        ----------
        obs_id : str or None, optional
            The observation ID to load. If `None`, loads whatever data is pointed to by `prepath`.
        event_dir : str or None, optional
            Name of the directory containing the event files. Default is event_l2
        energy_cut : tuple of float, optional
            Event energy range in keV. Default is (2, 8).

        Returns
        -------
        List[IXPEData]
            A list of `IXPEData` objects for all three detectors.
        '''

        reasons = []
        for prepath in DATA_DIRECTORIES:
            try:
                return IXPEData.load_all_detectors_with_path(prepath, obs_id, event_dir)
            except:
                reasons.append(sys.exc_info())
            
        # Could not find the file. Print out diagnostic information
        for (prepath, reason) in zip(DATA_DIRECTORIES, reasons):
            logger.warning(f"Checking {prepath}: {reason[1]}")
        raise Exception(f"Could not find any observations with ID {obs_id}")
    
    def load_all_detectors_with_path(prepath, obs_id=None, event_dir="event_l2"):
        '''
        Load all detectors from a specific directory, without using the default directories
        stored in the LeakageLib settings file.

        Parameters
        ----------
        prepath : str, optional
            If `obs_id` is not specified, `prepath` points to the folder containing the data
            for this observation. If `obs_id` is specified, `prepath` should point to the
            superfolder containing the observation folder, which is assumed to be named `obs_id`.
        obs_id : str or None, optional
            Observation ID to load. If `None`, loads data pointed to by `prepath`.
        event_dir : str or None, optional
            Directory containing event files. Default (`None`) is interpreted as `event_l2`
            for Moments data, or `event_nn` for NN results, based on `source`.
        energy_cut : tuple of float, optional
            Event energy range in keV. Default is (2, 8).

        Returns
        -------
        List[IXPEData]
            Returns a list of `IXPEData` objects for all three detectors.
        '''

        if obs_id is not None:
            event_directory = f"{prepath}/{obs_id}/{event_dir}"
            hk_directory = f"{prepath}/{obs_id}/hk"
        else:
            event_directory = f"{prepath}/{obs_id}/{event_dir}"
            hk_directory = f"{prepath}/hk"

        if not os.path.exists(event_directory):
            raise Exception("Event directory not found")
        if not os.path.exists(hk_directory):
            raise Exception("HK directory not found")

        hks = [None, None, None]
        for f in os.listdir(hk_directory):
            if not f.endswith(".fits"): continue
            if not f.startswith("ixpe"): continue
            if not "att" in f: continue
            if obs_id is not None:
                if not obs_id in f: continue
            if "det1" in f:
                hks[0] = f"{hk_directory}/{f}"
            elif "det2" in f:
                hks[1] = f"{hk_directory}/{f}"
            elif "det3" in f:
                hks[2] = f"{hk_directory}/{f}"

        detectors = []
        for f in os.listdir(event_directory):
            if not f.endswith(".fits"): continue
            if not f.startswith("ixpe"): continue
            if "det1" in f:
                i = 0
            elif "det2" in f:
                i = 1
            elif "det3" in f:
                i = 2
            else: continue

            if hks[i] is None:
                raise Exception(f"Could not find the housekeeping file for DU {i+1}. Did you download and unzip the housekeeping files?")
            detectors.append(IXPEData((f"{event_directory}/{f}", hks[i])))

        logger.info("Successfully loaded files")
        for data in detectors:
            logger.info(f"\t{data.filename}")

        return detectors

    def __init__(self, file_names):
        """
        Load the data from a single IXPE file.

        Parameters
        ----------
        source : Source
            A `Source` object used to set the size of the `IXPEData` images. The actual
            source flux is not read; only whether the source is NN or Mom and the shape
            of the image.
        file_names : tuple of str
            Tuple `(event_name, hk_name)` with paths to the event file and hk file.
        energy_cut : tuple of float, optional
            Event energy range in keV. Default is (2, 8).

        Returns
        -------
        IXPEData
            The detector data for the given file.
        """

        with fits.open(file_names[0]) as hdul:
            events = hdul[1].data
            self.det = int(hdul[1].header["DETNAM"][2:])
            if "OBS_ID" in hdul[1].header:
                self.obs_id = hdul[1].header['OBS_ID']
            else:
                self.obs_id = "None"
            if type(self.obs_id) == int:
                self.obs_id = f"{self.obs_id:08d}"

            self.use_nn = "W_NN" in hdul[1].columns.names
            self.use_nn_energies = self.use_nn
            if "NN_ENERGIES" in hdul[1].header and hdul[1].header["NN_ENERGIES"] == "F":
                self.use_nn_energies = False

            self.evt_xs = events["X"].astype(np.float64) * IXPE_PIXEL_SIZE
            self.evt_ys = events["Y"].astype(np.float64) * IXPE_PIXEL_SIZE
            self.evt_qs = events["Q"].astype(np.float64)
            self.evt_us = events["U"].astype(np.float64)
            self.evt_energies = 0.02 + events["PI"].astype(np.float64) * 0.04
            self.evt_pis = events["PI"].astype(int)
            self.evt_times = events["TIME"].astype(np.float64)
            self.evt_exposures = np.ones_like(self.evt_xs)
            self.evt_vigns = np.ones_like(self.evt_xs)
            if "BG_PROB" in events.columns.names:
                self.evt_bg_chars = events["BG_PROB"]
            else:
                self.evt_bg_chars = np.zeros(len(self.evt_xs))
            if self.use_nn:
                self.evt_mus = get_nn_modf(self.evt_energies)
            else:
                self.evt_mus = get_mom_modf(self.evt_energies)
            if "W_NN" in events.columns.names:
                self.evt_ws = events["W_NN"].astype(np.float64)
            else:
                self.evt_ws = events["W_MOM"].astype(np.float64)

        # Extract orientation from the housekeeping file too
        with fits.open(file_names[1]) as hdu:
            self.rotation = float(hdu[0].header['PADYN'])

        self.expmap = None

        self.filename = file_names[0]
        self.hk_filename = file_names[1]
        self.offsets = np.zeros(2)

        self._apply_vignetting()
        self._weight_nn()
        try:
            self.load_expmap()
        except:
            # Could not find exposure map
            pass
    
    def __len__(self):
        return len(self.evt_xs)

    def _apply_vignetting(self):
        center = 300*IXPE_PIXEL_SIZE
        off_axis_arcmin = np.sqrt((self.evt_xs - self.offsets[0] - center)**2 + (self.evt_ys - self.offsets[1] - center)**2) / 60
        vign = load_vign(du_id=self.det)
        self.evt_vigns = vign(self.evt_energies, off_axis_arcmin)

    def _weight_nn(self):
        """
        If this is an NN data set, scrap the energy-dependent modulation factor and instead treat
        the event-by-event weights as the modulation factor. This function will then set all the
        weights to one to avoid double counting them. If this is a Mom data set, this function
        will do nothing.
        """
        if self.use_nn:
            self.evt_mus = np.copy(self.evt_ws)
            self.evt_ws = np.ones_like(self.evt_ws)

    def retain(self, mask):
        """
        Retain all events according to a boolean mask, removing the rest.

        Parameters
        ----------
        mask : array-like of bool
            Mask where `True` indicates the event should be kept.
        """
        if len(mask) != len(self.evt_xs):
            raise Exception(f"The mask you provided (length {len(mask)}) does not have the same length as the data (length {len(self.evt_xs)})")
        self.evt_xs = self.evt_xs[mask]
        self.evt_ys = self.evt_ys[mask]
        self.evt_qs = self.evt_qs[mask]
        self.evt_us = self.evt_us[mask]
        self.evt_vigns = self.evt_vigns[mask]
        self.evt_energies = self.evt_energies[mask]
        self.evt_pis = self.evt_pis[mask]
        self.evt_exposures = self.evt_exposures[mask]
        self.evt_ws = self.evt_ws[mask]
        self.evt_mus = self.evt_mus[mask]
        self.evt_bg_chars = self.evt_bg_chars[mask]
        self.evt_times = self.evt_times[mask]

    def retain_region(self, regfile, exclude=False):
        """
        Cut all events according to a region file.

        Parameters
        ----------
        regfile : str
            Region file containing a single region, CIAO formatted, in physical coordinates.
        exclude : bool, optional
            If `True`, all events in the region will be removed. Otherwise, they will be kept.

        Returns
        -------
        Region
            The resulting region object.

        Notes
        -----
        Uses an approximate conversion between xy and RA/Dec, which may be slightly
        inaccurate off-axis.
        """

        region = Region(regfile, assert_format="image")
        mask = region.contains((self.evt_xs - self.offsets[0])/IXPE_PIXEL_SIZE, (self.evt_ys - self.offsets[1])/IXPE_PIXEL_SIZE)
        if exclude:
            mask = ~mask

        if np.sum(mask) == 0:
            logger.warning("Cut region to size zero")

        self.retain(mask)
        return region

    def explicit_center(self, x, y):
        """
        Recenter the dataset by offsets x, y in pixels
        """
        self.evt_xs -= x * IXPE_PIXEL_SIZE
        self.evt_ys -= y * IXPE_PIXEL_SIZE
        self.offsets -= (x*IXPE_PIXEL_SIZE, y*IXPE_PIXEL_SIZE)

    def centroid_center(self):
        """
        Recenter the dataset such that the event centroid is at (0, 0)
        """
        centroid = (np.nanmean(self.evt_xs), np.nanmean(self.evt_ys))
        self.evt_xs -= centroid[0]
        self.evt_ys -= centroid[1]
        self.offsets -= centroid

    def iterative_centroid_center(self):
        """
        Recenter the dataset such that the centroid of the core events is in the center. Do this by iteratively zooming in, so that the final center is set by the core of the PSF, not events in the wings. Yields more accurate results.
        """
        self.explicit_center(300, 300)

        for radius in [400, 100, 30]:
            mask = self.evt_xs**2 + self.evt_ys**2 < radius**2
            center = np.median([self.evt_xs[mask], self.evt_ys[mask]], axis=1)
            self.offsets -= center
            self.evt_xs -= center[0]
            self.evt_ys -= center[1]

    def get_antirotation_matrix(self):
        '''
        Get a 2D rotation matrix which reverses the detector's rotation
        '''
        return np.array([
            [np.cos(self.rotation), np.sin(self.rotation)],
            [np.sin(-self.rotation), np.cos(self.rotation)]
        ])

    def get_stokes_antirotation_matrix(self):
        '''
        Get a 2D rotation matrix which reverses the stokes's rotation
        '''
        return np.array([
            [np.cos(2*self.rotation), np.sin(2*self.rotation)],
            [np.sin(-2*self.rotation), np.cos(2*self.rotation)]
        ])

    def load_expmap(self, filename=None, offset=(0,0)):
        """
        Load the exposure map and evaluate it for every pixel in the image.

        Parameters
        ----------
        filename : str, optional
            Path to the exposure map. If `None`, the map is assumed to lie in the `auxil`
            folder, which is expected to be in the same directory as the hk and `event_l2` folders.
        offset : tuple of float, optional
            Exposure map offset in arcseconds.

        Notes
        -----
        This function should not be run after the image has been centered.
        """
        if filename is None:
            event_l2_folder = "/".join(self.filename.split("/")[:-1])
            auxil_folder = f"{event_l2_folder}/../auxil"
            if not os.path.exists(auxil_folder):
                raise Exception(f"Could not find the exposure map files in {auxil_folder}. Please use the filename argument to provide an explicit file")
            for f in os.listdir(auxil_folder):
                if "expmap2" not in f: continue
                if str(self.obs_id) not in f: continue
                if f"det{self.det}" not in f: continue
                filename = f"{auxil_folder}/{f}"
            if filename is None:
                raise Exception(f"Could not find an exposure map file in {auxil_folder}. Please use the filename argument to provide an explicit file")
        
        with fits.open(filename) as hdul:
            image = hdul[0].data
            wcs = WCS(hdul[0].header)
            upper_left, lower_right = wcs.all_pix2world([(0, 0), (image.shape[1]-1, image.shape[0]-1)], 0)
            ras = np.linspace(upper_left[0], lower_right[0], image.shape[0])
            decs = np.linspace(upper_left[1], lower_right[1], image.shape[1])

        with fits.open(self.filename) as hdul:
            colx = hdul[1].columns["X"]
            coly = hdul[1].columns["Y"]
            stretch = np.cos(coly.coord_ref_value * np.pi / 180)
            xs = ((ras - colx.coord_ref_value) / colx.coord_inc * stretch + colx.coord_ref_point) * IXPE_PIXEL_SIZE
            ys = ((decs - coly.coord_ref_value) / coly.coord_inc + coly.coord_ref_point) * IXPE_PIXEL_SIZE

        self.expmap = RegularGridInterpolator((xs, ys), np.transpose(image), bounds_error=False, fill_value=0)
        self.evt_exposures = self.expmap(self.evt_xs-offset[0], self.evt_ys-offset[0])

    def get_particle_flux_estimate(self):
        """
        Estimate the fraction of events in the dataset that are particles. This works by performing a mini-fit, neglecting all spatial and polarization information and just maximizing the likelihood for the particle flux given the observed particle characters.
        """
        bg_chars = np.clip(self.evt_bg_chars, 1e-5, 1 - 1e-5)
        term = (1 - bg_chars) / (2*bg_chars - 1)
        def solve_func(ps):
            return np.sum(1 / (term + ps))
        result = fsolve(solve_func, [0.5])
        return result[0]