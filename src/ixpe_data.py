import numpy as np
from astropy.io import fits
import os
from scipy.linalg import pinvh

from .spectrum import Spectrum
from .settings import DATA_DIRECTORIES

IXPE_PIXEL_SIZE = 2.6 # arcsec
CENTROID_CENTER = True

class IXPEData:
    def load_all_detectors(source, obs_id, energy_cut=(2, 8), time_cut_frac=None, weight_image=False):
        '''Load all detectors
        
        ## Arguments:
        - source: a Source object used to set the size of the IXPEData images. The actual source flux is not read, just the shape of the image.
        - obs_id: the observation ID to load. Searches in all the directories listed in settings.DATA_DIRECTORIES for these files.
        - use_nn: Set to true to use NN data
        - energy_cut: Event energy cut in keV. Default 2--8.
        - time_cut_frac: If not none, must be a tuple (l, h). Only events with time between the lth and hth quantiles will be used.

        ## Returns:
        A list of IXPEData objects for all three detectors. 
        '''
        for prepath in DATA_DIRECTORIES:
            result = IXPEData.load_all_detectors_with_path(prepath, source, obs_id, energy_cut, time_cut_frac, weight_image)
            if result is not None:
                return result
        raise Exception(f"Could not find any observations with ID {obs_id}")
    

    def load_all_detectors_with_path(prepath, source, obs_id, energy_cut=(2, 8), time_cut_frac=None, weight_image=False):
        '''Load all detectors
        
        ## Arguments:
        - prepath: The path to the data directory, either including or excluding the directory named after the observation ID.
        - source: a Source object used to set the size of the IXPEData images. The actual source flux is not read, just the shape of the image.
        - obs_id: the observation ID to load. Searches in all the directories listed in settings.DATA_DIRECTORIES for these files.
        - use_nn: Set to true to use NN data
        - energy_cut: Event energy cut in keV. Default 2--8.
        - time_cut_frac: If not none, must be a tuple (l, h). Only events with time between the lth and hth quantiles will be used.

        ## Returns:
        A list of IXPEData objects for all three detectors. 
        '''
        if len(obs_id) == 8:
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

        if not os.path.exists(hk_directory):
            return None

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
            if not "det2" in f: continue
            detectors[1] = IXPEData(source, (f"{event_directory}/{f}", hks[1]), energy_cut, time_cut_frac=time_cut_frac, weight_image=weight_image)

        det2_offset = detectors[1].total_shift

        for f in os.listdir(event_directory):
            if not f.endswith(".fits"): continue
            if not f.startswith("ixpe"): continue
            if "det1" in f:
                i = 0
            elif "det3" in f:
                i = 2
            else: continue
            detectors[i] = IXPEData(source, (f"{event_directory}/{f}", hks[i]), energy_cut, det2_offset, time_cut_frac=time_cut_frac, weight_image=weight_image)

        return detectors
    
    def get_net_polarization(data_list):
        '''Get the net polarization of all the Q and U in all three detectors image.'''

        polarizations = []
        for ixpe_data in data_list:
            polarization = ixpe_data.get_detector_polarization()
            polarizations.append(polarization)
        return np.mean(polarizations, axis=0)

    def __init__(self, source, file_name, energy_cut=(2, 8), det2_offset=None, time_cut_frac=None, weight_image=False):
        '''Make the IXPE data image for one detector.'''
        
        self.arcsec_per_unit = IXPE_PIXEL_SIZE
        with fits.open(file_name[0], memmap=False) as hdul:
            events = hdul[1].data
            self.det = int(hdul[1].header["DETNAM"][2:])
            x_center = events.columns["X"].coord_ref_point
            y_center = events.columns["Y"].coord_ref_point

            energies = 0.02 + events["PI"] * 0.04
            energy_mask = (energy_cut[0] <= energies) & (energies < energy_cut[1])
            events = events[energy_mask]

            times = events["TIME"]
            if time_cut_frac is not None:
                min_time = np.nanquantile(times, time_cut_frac[0])
                max_time = np.nanquantile(times, time_cut_frac[1])
                time_mask = (min_time < times) & (times < max_time)
                events = events[time_mask]

        # Extract orientation from the housekeeping file too
        with fits.open(file_name[1], memmap=False) as hdu:
            self.rotation = hdu[0].header['PADYN']

        self.pixels_per_row = len(source.source)
        self.pixel_size = source.pixel_size#arcsec

        self.pos_cut = self.pixels_per_row / 2 * self.pixel_size # Edge of the image in arcsec
        self.pixel_edges = np.arange(self.pixels_per_row + 1, dtype=float) * (self.pixel_size) # arcsec
        self.pixel_edges -= np.max(self.pixel_edges) / 2
        self.pixel_centers = self.pixel_edges[:-1] + (self.pixel_edges[1] - self.pixel_edges[0]) / 2
        self.counts = len(events)

        # Get the spectrum
        energies = 0.02 + np.arange(np.max(events["PI"]) + 1) * 0.04
        binned_counts = np.zeros_like(energies)
        binned_weights = np.zeros_like(energies)
        if "nn.fits" in file_name[0]:
            weights = events["W_NN"]
        else:
            weights = events["W_MOM"]
        for pi, weight in zip(events["PI"], weights):
            binned_counts[pi] += 1
            binned_weights[pi] += weight

        self.spectrum = Spectrum(energies, binned_counts, binned_weights / np.maximum(binned_counts, 1))

        # Extract the data
        total_shift = np.zeros(2)
        poses = np.array((events["X"], events["Y"])).transpose() * self.arcsec_per_unit

        if CENTROID_CENTER or self.det == 2:
            total_shift += np.nanmedian(poses, axis=0)
            poses -= np.nanmedian(poses, axis=0)

            image_mask = (np.abs(poses[:,0]) < self.pos_cut) & \
                (np.abs(poses[:,1]) < self.pos_cut)
            total_shift += np.nanmedian(poses[image_mask,:], axis=0)
            poses -= np.nanmedian(poses[image_mask,:], axis=0)
            

            # Zoom in and recenter again
            core_mask = np.sqrt(poses[:,0]**2 + poses[:,1]**2) < self.pos_cut / 3
            total_shift += np.nanmedian(poses[core_mask], axis=0)
            poses -= np.nanmedian(poses[core_mask], axis=0)

            self.total_shift = (total_shift - np.array([x_center, y_center]) * self.arcsec_per_unit)
        else:
            if det2_offset is None:
                raise Exception("If you don't use the centroid centering option, you must only load data sets as a group so that the pointing error of the telescope can be determined.")
            poses -= np.array([x_center, y_center]) * self.arcsec_per_unit + det2_offset

        # Compute the flux of really distant particles
        lower_radius = 120 # 2 arcmin
        upper_radius = 180 # 3 arcmin
        area = np.pi * (upper_radius**2 - lower_radius**2)
        dists = np.sqrt(poses[:,0]**2 + poses[:,1]**2)
        self.bg_flux_annulus = float(np.sum((lower_radius < dists) & (dists < upper_radius))) / area
        # By default, this flux is not subtracted away, but you can if you want to 

        # Compute the actual mask for the image
        image_mask = (np.abs(poses[:,0]) < self.pos_cut) & \
            (np.abs(poses[:,1]) < self.pos_cut)
        masked_events = events[image_mask]
        poses = poses[image_mask]
        weights = weights[image_mask]

        xi = np.digitize(poses[:,1], self.pixel_edges[1:])
        yi = np.digitize(poses[:,0], self.pixel_edges[1:])

        shape = (self.pixels_per_row, self.pixels_per_row)
        self.i = np.zeros(shape)
        self.q = np.zeros(shape)
        self.u = np.zeros(shape)
        self.cov_inv = np.zeros((shape[0], shape[1], 2, 2))

        for x_index in range(self.pixels_per_row):
            for y_index in range(self.pixels_per_row):
                mask = (xi == x_index) & (yi == y_index)
                if weight_image:
                    pixel_i = np.sum(weights[mask])
                    qs = masked_events["Q"][mask] * weights[mask]
                    us = masked_events["U"][mask] * weights[mask]
                else:
                    pixel_i = np.sum(mask)
                    qs = masked_events["Q"][mask]
                    us = masked_events["U"][mask]
                mean_q = np.mean(qs)
                mean_u = np.mean(us)

                # Method 1: the cov matrix found in the Kislat paper
                cov = pixel_i * np.array([
                    [2-mean_q*mean_q, -mean_q*mean_u],
                    [-mean_q*mean_u, 2-mean_u*mean_u]
                ])

                # Method 2: the cov matrix I derived. But it can be multiplied by an arbitrary factor.
                # cov = (4 / np.sum(mask)) * np.mean([
                #     [us*us, -qs*us],
                #     [-qs*us, qs*qs]
                # ], axis=-1)

                # Method 3: the cov matrix of the data
                # cov = np.cov(qs, us) * pixel_i

                if np.linalg.det(cov) < 0 or np.any(np.isnan(cov)) or np.sum(mask) == 0:
                    # Bail on this pixel
                    self.i[x_index,y_index] = 0#np.nan
                    self.q[x_index,y_index] = 0#np.nan
                    self.u[x_index,y_index] = 0#np.nan
                    self.cov_inv[x_index,y_index] = np.nan
                else:
                    self.i[x_index,y_index] = pixel_i
                    self.q[x_index,y_index] = np.sum(qs)
                    self.u[x_index,y_index] = np.sum(us)
                    self.cov_inv[x_index,y_index] = pinvh(cov)

    def get_antirotation_matrix(self):
        '''Get a 2D rotation matrix which reverses the detector's rotation'''
        return np.array([
            [np.cos(self.rotation), np.sin(self.rotation)],
            [np.sin(-self.rotation), np.cos(self.rotation)]
        ])
    
    def background_subtract(self):
        '''Subtract the background flux received in a 2-3 arcsec annulus around the point source. Don't subtract any polarization.'''
        self.i -= self.bg_flux_annulus * self.pixel_size**2

    def get_detector_polarization(self):
        '''Get the polarization of a single IXPEdata by averaging over the entire image'''
        mask = self.i > 500
        normalized_q = self.q / self.i
        normalized_u = self.u / self.i
        return np.array([np.nanmean(normalized_q[mask]), np.nanmean(normalized_u[mask])])
