Polarization fitting
====================

`LeakageLib` polarimetric fitting has two main advantages over other X-ray polarization fitting methods:

1. It delivers precise fit results, enabled by advanced weights that help distinguish source events from background events.
2. It can fit complex fields, such as those that contain extended sources, overlapping sources, sources with EVPA and PD sweeps over time or space, and sources with different spectra.

It was shown in `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_ that LeakageLib delivers at least a 2x decrease in the QU uncertainty contour area compared to the simplest polarimetric extraction method; that is, it delivers uncertainties equivalent to doubling the amount of available data. Later analyses have sometimes achieved even larger gains.

Examples: Point source fitting
------------------------------

The below Jupyter notebooks give examples of using LeakageLib for polarimetric fitting of point sources. The first notebook performs a basic fit using simulated data. Start there if you're new to LeakageLib.

The second through fourth notebooks treat real data from the accreting pulsar GX 301-2. They describe how to prepare the data for fitting and then fit a simple constant polarization model.

The remaining notebooks demonstrate some more advanced features, like temporal weighting and MCMC fitting.

.. toctree::
    :maxdepth: 2
    :caption: Point source Jupyter notebook examples

    examples/fit-point-source.ipynb
    examples/fit-prepare-point-source.ipynb
    examples/fit-point-source-real.ipynb
    examples/fit-point-source-real-particle-weights.ipynb
    examples/fit-nn.ipynb
    examples/fit-time-varying.ipynb
    examples/fit-mcmc.ipynb

Examples: Extended source fitting
---------------------------------

LeakageLib's spatial weights allow the user to fit for polarizations of extended sources. Spatial weights require knowledge of the original location of IXPE photons. If the source is point like, you can use the in-built :py:meth:`leakagelib.Source.delta` function to create such a source. But for extended sources, you need an externally generated flux map. LeakageLib comes shipped with a command line interface called `leakagelib_cxo`. This program can generate the sources necessary to do LeakageLib spatial weighting from Chandra event files by rescaling the Chandra events to the IXPE band. Once you have created the rescaled image, it is very simple to import it into LeakageLib using the :py:meth:`leakagelib.Source.load_file` method.

The first example below demonstrates extended source fitting with a disk-like, simulated source. The remaining notebooks prepare and fit to real data from the Crab pulsar and PWN.

.. toctree::
    :maxdepth: 2
    :caption: Extended source Jupyter notebook examples

    examples/fit-extended.ipynb
    examples/fit-prepare-extended.ipynb
    examples/fit-extended-real.ipynb

How LeakageLib fitting works
----------------------------

LeakageLib determines the probability for each event to originate from each possible source, including background sources. It uses these probability "weights" to constrain the polarization of each source. Events which are likely to constrain one particular source are up-weighted (e.g. events that are spatially coincident or have the right energy), and less constraining events (e.g. events with poor modulation factor) do not affect the fit. The weights LeakageLib uses are

* **Spatial weights**, which help remove background and separate overlapping sources. LeakageLib's method uses weights in a way which avoids pollution from polarization leakage.
* **Polarimetric weights**, i.e. individual modulation factors for each event. The neural net designed by Lawrence Peirson provides these per-event modulation factors.
* **Particle weights**, which de-weight IXPE events which are morphologically similar to particles. Most IXPE analyses cut particles based on level 1 event data, making a trade-off between cutting too few particles or too many photons. De-weighting avoids this trade-off, which increases precision.
* **Phase/time weights**, if some of the sources are pulsed or time variable.
* **Energy weights**, which use event energies to distinguish between sources with different spectra. LeakageLib currently only works if the spectrum is already known; full spectro-polarimetric fitting (i.e. simultaneously fitting spectrum parameters and polarization parameters) is not yet implemented.

These weights are incorporated using a maximum likelihood technique. For convenience, we derive the basic likelihood here, but the user should refer to `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_ for details.

Derivation of the LeakageLib likelihood
---------------------------------------

The likelihood is defined as the probability of detecting the observed data given the fit parameters. Here, the "data" consist of the EVPA :math:`\psi_i`, position :math:`\mathbf r_i`, energy :math:`E_i`, and time :math:`t_i` of each event. :math:`i` refers to the index of the event. The fit parameters are the sources' PD and EVPA :math:`\Pi_s` and :math:`\Psi_s`. :math:`s` refers to the source index. The flux of each source :math:`f_s` is also a fit parameter. It is convenient to units where the total flux is 1. I.e. :math:`f_s` is the fraction of the data that came from source :math:`s`. We can therefore write the likelihood mathematically as 

.. math::
    L = P(\{\psi_i, \mathbf r_i, E_i, t_i\} | \{\Pi_s, \Psi_s, f_s\}).

Since each event is independent, we can use the identity :math:`P(X,Y) = P(X)P(Y)` for independent random variables :math:`X` and :math:`Y`, writing

.. math::
    L = \prod_i P(\psi_i, \mathbf r_i, E_i, t_i | \{\Pi_s, \Psi_s, f_s\}).

Here we can use another identity: :math:`P(X) = \sum_Y P(X|Y) P(Y)`, where the sum is over all possible values of :math:`Y`. We apply this where :math:`Y` represents the originating source :math:`s` of the photon. Then :math:`P(s)` is :math:`f_s` and we write :math:`P(X | s)` as :math:`P_s(X)` for convenience.

.. math::
    L = \prod_i \sum_s f_s P_s(\psi_i, \mathbf r_i, E_i, t_i | \Pi_s, \Psi_s).

The innermost term represents the probability that source :math:`s` emits an event with the observed properties of event :math:`i`, which is relatively easy to model. Let us assume that the source has constant polarization for simplicity, so that time is independent of other variables. So is energy, after spurious modulation and vignetting correction. Therefore,

.. math::
    L = \prod_i \sum_s f_s P_s(\psi_i, \mathbf r_i | \Pi_s, \Psi_s) P_s(E_i) P_s(t_i).

:math:`P_s(E_i)` is the source spectrum, and :math:`P_s(t_i)` is the source light curve.

Unfortunately, :math:`\psi_i` and :math:`\mathbf r_i` are not independent due to polarization leakage. The best we can write is :math:`P_s(\psi_i, \mathbf r_i | \Pi_s, \Psi_s) = P_s(\mathbf r_i | \psi_i) P_s(\psi_i | \Pi_s, \Psi_s)`, where :math:`P_s(\mathbf r_i | \psi_i)` can be derived from the polarization leakage theory, and :math:`P_s(\psi_i | \Pi_s, \Psi_s)` is the standard distribution

.. math::
    P_s(\psi_i | \Pi_s, \Psi_s) = \frac{1}{2 \pi} \left[1 + \Pi_s \mu_i \cos (2[\psi_i - \Psi_s])\right].

Combining these terms,

.. math::
    L = \prod_i \sum_s f_s P_s(\mathbf r_i | \psi_i) P_s(\psi_i | \Pi_s, \Psi_s) P_s(E_i) P_s(t_i)

is the (simplified) LeakageLib likelihood. These terms represent respectively the source flux, spatial weights, polarimetric weights, energy weights, and time/phase weights. The source flux can be converted to particle weights with further considerations given in Dinsmore & Romani 2025. I say "simplified" because LeakageLib is also capable of treating sources with time-varying polarization, which involves slight modifications to this likelihood.

Caveats
-------

There are several pitfalls that any user of LeakageLib should check before publishing fits conducted with `LeakageLib`.

* **The initial parameter guesses need to be accurate for fits to be repeatable**. When fitting with many parameters, it is easy to be trapped in a local minimum. To avoid this, the user should start using a simple fit and slowly add more parameters (e.g. background polarization, different polarizations in each detector, etc.), adjusting the initial parameter guesses to align with the simpler fit results using the :py:meth:`leakagelib.FitSettings.set_initial_qu` and :py:meth:`leakagelib.FitSettings.set_initial_flux` functions. If these are not used, the results may be inaccurate even if the fit succeeds.
* **All sources need to be aligned with the data**. For the spatial weights to be effective, the IXPE data and sources need to be aligned with each other. This requires some manual work, e.g. centering the target if observing a point source so that the :py:meth:`leakagelib.Source.delta` point source, which lies in the center of the image, is correct. If using spatial maps from Chandra, the Chandra and IXPE WCSs need to be aligned.
* **All flux in the region of interest must included in a fit component**. If you are fitting to an observation with multiple bright sources, each source requires a component. This is true even for sources whose polarization you are not interested in. If a component is not modeled, the source will be treated as background, biasing the flux and polarization of the background fit. Alternatively, you may  exclude the source from the fit by cutting the surrounding area out of the region of interest (ROI), in which case it need not be modeled.
* **Failure to use spectral weights gives small amounts of bias**. This is true of any MLE method, as pointed out in section 2.4 of `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_. The bias is typically at the ~5% level, so it matters only for fairly well-measured sources. It is best addressed by creating accurate or approximate spatial weights for each source and background component.
* **All spatial cuts must be reflected in the ROI**. If the region of interest (ROI) does not include any spatially cut regions, the normalization of the spatial probability distribution will be incorrect. This will bias the spatial weights and corrupt the fit results. If you apply spatial cuts to the data, you must create a custom ROI that is zero-valued wherever events were cut.
* **When using spectral weights, non-contiguous energy cuts need to be noted**. For example, if 4-6 keV events are removed and spatial weighting is used, the user must tell LeakageLib using the `duty_cycle` argument of the `set_spectrum` function (see documentation of :py:meth:`leakagelib.FitSettings.set_spectrum`). Otherwise, the normalization of the energy weights is wrong.
* **When using temporal/phase weights, non-contiguous time cuts need to be noted**. For example, if only events in some range of phases for a pulsating source are used, the user must tell LeakageLib using the `duty_cycle` argument of the `set_lightcurve` function (see documentation of :py:meth:`leakagelib.FitSettings.set_lightcurve`). Otherwise, the normalization of the phase weights is wrong.