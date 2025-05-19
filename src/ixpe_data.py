import numpy as np
import os
from scipy.linalg import pinvh
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
from astropy.wcs import WCS
from .spectrum import Spectrum
from .region import make_region
from .settings import DATA_DIRECTORIES
from .modulation import *

IXPE_PIXEL_SIZE = 2.6 # arcsec

class IXPEData:
    """A class containing the data of a single pointing and a single detector.
    # Fields
    * General information
        - det lists the detector (1, 2, 3)
        - filename contains the FITS file representing this data set
        - 
    * Events
        - The fields evt_*, where *=xs, ys, qs, us, energies, pis, times, ws, mus, and bg_probs, contain the properties of the individual events. Positions are measured in arcsec; q and u are the IXPE-standard 2cos(psi) and 2sin(psi). To cut them, use the IXPEData.cut(mask) or IXPEData.cut_region(reg_file) functions. Do not manually edit these fields, as this may cause the IXPEData to go out of sync with itself.
    * Images
        - The fields i, q, u, n, w2, and cov_inv contain images of the observation, constructed with the same pixels as the Source provided upon initialization. These fields will exist if you constructed IXPEData with bin=True. If you need to recreate these images, call IXPEData.bin_data().
    """

    def load_all_detectors(source, obs_id, energy_cut=(2, 8), time_cut_frac=None, weight_image=False, bin=True):
        '''Load all detectors
        
        ## Arguments:
        - source: a Source object used to set the size of the IXPEData images. The actual source flux is not read, just whether the source is NN or Mom and the shape of the image. If you do not wish to bin the data, you can pass a Source.no_image
        - obs_id: the observation ID to load. Searches in all the directories listed in settings.DATA_DIRECTORIES for these files.
        - use_nn: Set to true to use NN data
        - energy_cut: Event energy cut in keV. Default 2--8.
        - time_cut_frac: If not None, must be a tuple (l, h). Only events with time between the lth and hth quantiles will be used.
        - weight_image: Set to True to weight the Q and U image by weights
        - bin: Set to False if you don't wish to bin the data. If you passed a source created with Source.no_image, then the bin argument will be ignored and bins will not be produced.

        ## Returns:
        A list of IXPEData objects for all three detectors. 
        '''

        reasons = []
        for prepath in DATA_DIRECTORIES:
            result, reason = IXPEData.load_all_detectors_with_path(source, prepath, obs_id, energy_cut, time_cut_frac, weight_image, bin)
            reasons.append(reason)
            if result is not None:
                return result
            
        # Could not find the file. Print out diagnostic information
        for (prepath, reason) in zip(DATA_DIRECTORIES, reasons):
            print(f"Checking {prepath}: {reason}")
        raise Exception(f"Could not find any observations with ID {obs_id}")
    

    def load_all_detectors_with_path(source, prepath, obs_id=None, energy_cut=(2, 8), time_cut_frac=None, weight_image=False, bin=True):
        '''Load all detectors
        
        ## Arguments:
        - source: a Source object used to set the size of the IXPEData images. The actual source flux is not read, just whether the source is NN or Mom and the shape of the image. If you do not wish to bin the data, you can pass a Source.no_image
        - prepath: The path to the data directory. If obs_is not specified, this should point to the folder that contains all observation data. If obs_id is specified, this should point to the superfolder.
        - obs_id: the observation ID to load. Set to None to load whatever data is pointed to by prepath.
        - energy_cut: Event energy cut in keV. Default 2--8.
        - time_cut_frac: If not None, must be a tuple (l, h). Only events with time between the lth and hth quantiles will be used.
        - weight_image: Set to True to weight the Q and U image by weights
        - bin: Set to False if you don't wish to bin the data. If you passed a source created with Source.no_image, then the bin argument will be ignored and bins will not be produced.

        ## Returns:
        A list of IXPEData objects for all three detectors. 
        '''

        if obs_id is not None:
            if source.use_nn:
                event_directory = f"{prepath}/{obs_id}/event_nn"
            else:
                event_directory = f"{prepath}/{obs_id}/event_l2"
            hk_directory = f"{prepath}/{obs_id}/hk"
        else:
            if source.use_nn:
                event_directory = f"{prepath}/event_nn"
            else:
                event_directory = f"{prepath}/event_l2"
            hk_directory = f"{prepath}/hk"

        if not os.path.exists(event_directory):
            return (None, "Event directory not found")
        if not os.path.exists(hk_directory):
            return (None, "HK directory not found")

        hks = [None, None, None]
        for f in os.listdir(hk_directory):
            if not f.endswith(".fits"): continue
            if not f.startswith("ixpe"): continue
            if not "att" in f: continue
            if "det1" in f:
                hks[0] = f"{hk_directory}/{f}"
            if "det2" in f:
                hks[1] = f"{hk_directory}/{f}"
            if "det3" in f:
                hks[2] = f"{hk_directory}/{f}"

        for hk in hks:
            if hk is None:
                raise Exception("Could not find all the housekeeping files. Did you unzip them?")

        detectors = [None, None, None]
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
            detectors[i] = IXPEData(source, (f"{event_directory}/{f}", hks[i]), energy_cut, time_cut_frac=time_cut_frac, weight_image=weight_image, bin=bin)

        print("Successfully loaded files")
        for data in detectors:
            if data is None: continue
            print(f"\t{data.filename}")

        return (detectors, "")
    
    def get_net_polarization(data_list):
        '''Get the net polarization of all the Q and U in all three detectors image.'''

        polarizations = []
        for ixpe_data in data_list:
            polarization = ixpe_data.get_detector_polarization()
            polarizations.append(polarization)
        return np.mean(polarizations, axis=0)

    def __init__(self, source, file_name, energy_cut=(2, 8), time_cut_frac=None, weight_image=False, bin=True):
        '''Load all detectors
        
        ## Arguments:
        - source: a Source object used to set the size of the IXPEData images. The actual source flux is not read, just whether the source is NN or Mom and the shape of the image. If you do not wish to bin the data, you can pass a Source.no_image
        - file_name: A tuple (event_name, hk_name). `event_name` points to the event_file. `hk_name` points to the hk file.
        - energy_cut: Event energy cut in keV. Default 2--8.
        - time_cut_frac: If not None, must be a tuple (l, h). Only events with time between the lth and hth quantiles will be used.
        - weight_image: Set to True to weight the Q and U image by weights
        - bin: Set to False if you don't wish to bin the data. If you passed a source created with Source.no_image, then the bin argument will be ignored and bins will not be produced.

        ## Returns:
        The detector in question
        '''

        if not source.has_image:
            bin = False

        with fits.open(file_name[0]) as hdul:
            events = hdul[1].data
            self.det = int(hdul[1].header["DETNAM"][2:])
            self.obs_id = hdul[1].header["OBS_ID"]

            energies = 0.02 + events["PI"] * 0.04
            # energy mask
            self.mask = (energy_cut[0] <= energies) & (energies < energy_cut[1])

            if time_cut_frac is not None:
                times = events["TIME"]
                min_time = np.nanquantile(times, time_cut_frac[0])
                max_time = np.nanquantile(times, time_cut_frac[1])
                self.mask &= (min_time < times) & (times < max_time)
            events = events[self.mask]

            self.evt_xs = events["X"] * IXPE_PIXEL_SIZE
            self.evt_ys = events["Y"] * IXPE_PIXEL_SIZE
            self.evt_qs = events["Q"]
            self.evt_us = events["U"]
            self.evt_energies = 0.02 + events["PI"] * 0.04
            self.evt_pis = events["PI"]
            self.evt_times = events["TIME"]
            self.evt_exposures = np.ones_like(self.evt_xs)
            if "BG_PROB" in events.columns.names:
                self.evt_bg_probs = events["BG_PROB"]
            else:
                self.evt_bg_probs = np.zeros(len(self.evt_xs))
            if source.use_nn:
                self.evt_mus = get_nn_modf(self.evt_energies)
            else:
                self.evt_mus = get_mom_modf(self.evt_energies)
            if "nn.fits" in file_name[0]:
                self.evt_ws = events["W_NN"]
            else:
                self.evt_ws = events["W_MOM"]

        # Extract orientation from the housekeeping file too
        with fits.open(file_name[1]) as hdu:
            self.rotation = hdu[0].header['PADYN']

        self.bin = bin
        self.pixels_per_row = len(source.source)
        self.pixel_size = source.pixel_size#arcsec

        self.pos_cut = self.pixels_per_row / 2 * self.pixel_size # Edge of the image in arcsec
        self.pixel_edges = np.arange(self.pixels_per_row + 1, dtype=float) * (self.pixel_size) # arcsec
        self.pixel_edges -= np.max(self.pixel_edges) / 2
        self.pixel_centers = self.pixel_edges[:-1] + (self.pixel_edges[1] - self.pixel_edges[0]) / 2
        self.counts = len(events)
        self.weight_image = weight_image
        self.use_nn = source.use_nn
        self.filename = file_name[0]

        self.extract_spectrum()
        self.bin_data()

    def cut(self, mask):
        """Cut all events to a mask
        # Arguments:
        * mask: a mask where True indicates the event should be kept
        """
        current_indices = np.arange(len(self.mask)).astype(int)[self.mask]
        excluded_indices = current_indices[~mask]
        self.mask[excluded_indices] = False
        self.evt_xs = self.evt_xs[mask]
        self.evt_ys = self.evt_ys[mask]
        self.evt_qs = self.evt_qs[mask]
        self.evt_us = self.evt_us[mask]
        self.evt_energies = self.evt_energies[mask]
        self.evt_pis = self.evt_pis[mask]
        self.evt_exposures = self.evt_exposures[mask]
        self.evt_ws = self.evt_ws[mask]
        self.evt_mus = self.evt_mus[mask]
        self.evt_bg_probs = self.evt_bg_probs[mask]
        self.evt_times = self.evt_times[mask]
        self.extract_spectrum()
        self.bin_data()

    def cut_region(self, regfile, exclude=False):
        """
        Cut all events according to a region file
        # Arguments:
        * regfile: a region file containing a single region, ciao formatted, in physical coordinates.
        * exclude: if set to True, all events in the region will be cut. Otherwise, they will be kept. 

        WARNING: Uses an approximate conversion between xy and radec which will be slightly inaccurate off-axis,
        """

        region = make_region(regfile)
        mask = region.check_inside_absolute(self.evt_xs/IXPE_PIXEL_SIZE, self.evt_ys/IXPE_PIXEL_SIZE)
        if exclude:
            mask = ~mask

        if np.sum(mask) == 0:
            print("WARNING: Cut region to size zero")

        self.cut(mask)

    def extract_spectrum(self):
        energies = 0.02 + np.arange(np.max(self.evt_pis) + 1) * 0.04
        binned_counts = np.zeros_like(energies)
        binned_weights = np.zeros_like(energies)
        for pi, weight in zip(self.evt_pis, self.evt_ws):
            binned_counts[pi] += 1
            binned_weights[pi] += weight

        averaged_weights = binned_weights / np.maximum(binned_counts, 1)

        self.spectrum = Spectrum(energies, binned_counts, averaged_weights)

    def explicit_center(self, x, y):
        "Recenter the dataset to position x, y in pixels"
        self.evt_xs -= x * IXPE_PIXEL_SIZE
        self.evt_ys -= y * IXPE_PIXEL_SIZE
        self.bin_data()

    def centroid_center(self):
        "Recenter the dataset such that the event centroid is at (0, 0)"
        self.evt_xs -= np.nanmean(self.evt_xs)
        self.evt_ys -= np.nanmean(self.evt_ys)
        self.bin_data()

    def iterative_centroid_center(self):
        "Recenter the dataset such that the centroid of the core events is in the center"
        poses = np.array((self.evt_xs, self.evt_ys)).transpose()
        poses -= np.nanmedian(poses, axis=0)

        image_mask = (np.abs(poses[:,0]) < self.pos_cut) & \
            (np.abs(poses[:,1]) < self.pos_cut)
        poses -= np.nanmedian(poses[image_mask,:], axis=0)

        # Zoom in and recenter again
        core_mask = np.sqrt(poses[:,0]**2 + poses[:,1]**2) < self.pos_cut / 3
        poses -= np.nanmedian(poses[core_mask], axis=0)

        self.evt_xs = poses[:,0]
        self.evt_ys = poses[:,1]
        self.bin_data()

    def bin_data(self):
        if not self.bin: return
        poses = np.array((self.evt_xs, self.evt_ys)).transpose()
        qus = np.array((self.evt_qs, self.evt_us)).transpose()

        # Compute the actual mask for the image
        image_mask = (np.abs(poses[:,0]) < self.pos_cut) & \
            (np.abs(poses[:,1]) < self.pos_cut)
        poses = poses[image_mask]
        qus = qus[image_mask]
        weights = self.evt_ws[image_mask]

        xi = np.digitize(poses[:,1], self.pixel_edges[1:])
        yi = np.digitize(poses[:,0], self.pixel_edges[1:])

        shape = (self.pixels_per_row, self.pixels_per_row)
        self.i = np.zeros(shape)
        self.q = np.zeros(shape)
        self.u = np.zeros(shape)
        self.n = np.zeros(shape) # Unweighted i
        self.w2 = np.zeros(shape) # Weights squared
        self.cov_inv = np.zeros((shape[0], shape[1], 2, 2))

        total_mask = np.zeros_like(xi)

        for x_index in range(self.pixels_per_row):
            for y_index in range(self.pixels_per_row):
                mask = (xi == x_index) & (yi == y_index)
                total_mask += mask
                if self.weight_image:
                    pixel_i = np.sum(weights[mask])
                    qs = qus[mask,0] * weights[mask]
                    us = qus[mask,1] * weights[mask]
                else:
                    pixel_i = np.sum(mask)
                    qs = qus[mask,0]
                    us = qus[mask,1]
                mean_q = np.mean(qs)
                mean_u = np.mean(us)

                cov = pixel_i * np.array([
                    [2-mean_q*mean_q, -mean_q*mean_u],
                    [-mean_q*mean_u, 2-mean_u*mean_u]
                ])

                if np.sum(mask) == 0:
                    # Bail on this pixel
                    self.cov_inv[x_index,y_index] = np.nan
                else:
                    self.n[x_index,y_index] = np.sum(mask)
                    self.w2[x_index,y_index] = np.sum(weights[mask]**2)
                    self.i[x_index,y_index] = pixel_i
                    self.q[x_index,y_index] = np.sum(qs)
                    self.u[x_index,y_index] = np.sum(us)
                    self.cov_inv[x_index,y_index] = pinvh(cov)


    def antirotate_events(self):
        """
        Aligns the events so that up is in detector coords rather than north and saves them as evt_xs_antirot, evt_ys_antirot, evt_qs_antirot, evt_us_antirot. The original evt_xs and evt_ys and the image are unaffected.
        """
        self.evt_xs_antirot, self.evt_ys_antirot = np.einsum(
            "ij,ja->ia",
            self.get_antirotation_matrix(), 
            [self.evt_xs, self.evt_ys]
        )
        self.evt_qs_antirot, self.evt_us_antirot = np.einsum(
            "ij,ja->ia",
            self.get_stokes_antirotation_matrix(), 
            [self.evt_qs, self.evt_us]
        )

    def get_antirotation_matrix(self):
        '''Get a 2D rotation matrix which reverses the detector's rotation'''
        return np.array([
            [np.cos(self.rotation), np.sin(self.rotation)],
            [np.sin(-self.rotation), np.cos(self.rotation)]
        ])

    def get_stokes_antirotation_matrix(self):
        '''Get a 2D rotation matrix which reverses the stokes's rotation'''
        return np.array([
            [np.cos(2*self.rotation), np.sin(2*self.rotation)],
            [np.sin(-2*self.rotation), np.cos(2*self.rotation)]
        ])

    def get_detector_polarization(self):
        '''Get the polarization of a single IXPEdata by averaging over the entire image'''
        mask = self.i > 500
        normalized_q = self.q / self.i
        normalized_u = self.u / self.i
        return np.array([np.nanmean(normalized_q[mask]), np.nanmean(normalized_u[mask])])

    def load_expmap(self, filename=None):
        '''Loads the exposure map and evaluates it for every pixel in the image.
        # Arguments
        * filename: location of the exposure map. If none, the map is assumed to lie in the auxil folder, which is assumed to lie in the same directory as the hk and event_l2 folders.
        
        WARNING: This function should not be run after the image has been centered.
        '''
        if filename is None:
            event_l2_folder = "/".join(self.filename.split("/")[:-1])
            auxil_folder = f"{event_l2_folder}/../auxil"
            if not os.path.exists(auxil_folder):
                raise Exception(f"Could not find the exposure map files in {auxil_folder}. Please use the filename argument to provide an explicit location")
            for f in os.listdir(auxil_folder):
                if "expmap2" not in f: continue
                if str(self.obs_id) not in f: continue
                if f"det{self.det}" not in f: continue
                filename = f"{auxil_folder}/{f}"
        
        with fits.open(filename) as hdul:
            image = hdul[0].data
            wcs = WCS(hdul[0].header)
            upper_left, lower_right = wcs.all_pix2world([(0, 0), (image.shape[1]-1, image.shape[0]-1)], 0)
            ras = np.linspace(upper_left[0], lower_right[0], image.shape[0])
            decs = np.linspace(upper_left[1], lower_right[1], image.shape[1])
            expmap = RegularGridInterpolator((ras, decs), np.transpose(image))

        with fits.open(self.filename) as hdul:
            # Retrieve the connection between x/y and ra/dec
            colx = hdul[1].columns["X"]
            coly = hdul[1].columns["Y"]
            stretch = np.cos(coly.coord_ref_value * np.pi / 180)
            ras = (self.evt_xs / IXPE_PIXEL_SIZE - colx.coord_ref_point) * stretch * colx.coord_inc + colx.coord_ref_value
            decs = (self.evt_ys / IXPE_PIXEL_SIZE - coly.coord_ref_point) * coly.coord_inc + coly.coord_ref_value

        self.evt_exposures = expmap((ras, decs))
