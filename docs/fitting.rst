Polarization fitting
====================

`LeakageLib` polarimetric fitting has two main advantages over other X-ray polarization fitting methods:

1. It delivers precise and accurate polarization fit results, enabled by advanced weights that help distinguish source events from background events.
2. It can fit complex fields, that contain e.g. extended sources, overlapping sources, and sources with EVPA and PD sweeps.

It was shown in `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_ that LeakageLib delivers a 2x decrease in the QU uncertainty contour area compared to the simplest polarimetric extraction method; that is, it delivers uncertainties equivalent to doubling the amount of available data.

How LeakageLib fitting works
----------------------------

LeakageLib determines the probability for each event to originate from each possible source, including background sources. It uses these probability "weights" to constrain the polarization of each source. Events which are likely to constrain one particular source are up-weighted (e.g. events that are spatially coincident or have the right energy), and less constraining events (e.g. events with poor modulation factor) do not affect the fit. The weights `LeakageLib` uses are

* **Spatial weights**, which help remove background and separate overlapping sources. LeakageLib's method uses weights in a way which avoids pollution from polarization leakage.
* **Polarimetric weights**, i.e. individual modulation factors for each event. Most analyses treat modulation factor as a function of event energy, and properly using these provides a very large precision gain. The neural net designed by Lawrence Peirson provides more individualized estimates of these modulation factors for each event. These provide additional gain.
* **Particle weights**, which de-weight IXPE events which are morphologically similar to particles. Most IXPE analyses cut particles based on level 1 event data, making a trade-off between cutting too few particles or too many photons. De-weighting avoids this trade-off, which increases precision
* **Phase/time weights**, if some of the sources are pulsed or time variable.
* **Energy weights**, which use event energies to distinguish between sources with different spectra. These are useful if the spectrum is already known, but LeakageLib is not yet designed for full spectro-polarimetric fitting (i.e. simultaneously fiting spectrum parameters and polarization parameters).

These weights are incorporated using a maximum likelihood technique described in `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_.

Caveats
-------

There are several pitfalls that any user of LeakageLib should check before publishing fits conducted with `LeakageLib`.

* **The initial parameter guesses need to be accurate for fits to be repeatable**. When fitting with many parameters, it is easy to be trapped in a local minimum. To avoid this, the user should start using a simple fit and slowly add more parameters (e.g. background polarization, different polarizations in each detector, etc.), adjusting the initial parameter guesses to align with the simpler fit results using the :meth:`FitSettings.set_initial_qu` and :meth:`FitSettings.set_initial_flux` functions. If these are not used, the results may be inaccurate even if the fit succeeds.
* **Failure to use spectral weights gives small amounts of bias**. This is true of any MLE method, as pointed out in section 2.4 of `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_. The bias is typically at the ~5% level, so it matters only for fairly well-measured sources. It is best addressed by creating accurate or approximate spatial weights for each source and background component.
* **All spatial cuts must be reflected in the ROI**. If the region of interest (ROI) does not include any spatially cut regions, the normalization of the spatial probability distribution will be incorrect. This will bias the spatial weights and corrupt the fit results. If you apply spatial cuts to the data, you must create a custom ROI that is zero-valued wherever events were cut.
* **When using spectral weights, non-contiguous energy cuts need to be noted**. For example, if 4-6 keV events are removed and spatial weighting is used, the user must tell LeakageLib using the `duty_cycle` argument of the `set_spectrum` function (see documentation of :meth:`FitSettings.set_spectrum`). Otherwise, the normalization of the energy weights is wrong.
* **When using temporal/phase weights, non-contiguous time cuts need to be noted**. For example, if only events in some range of phases for a pulsating source are used, the user must tell LeakageLib using the `duty_cycle` argument of the `set_lightcurve` function (see documentation of :meth:`FitSettings.set_lightcurve`). Otherwise, the normalization of the phase weights is wrong.

Examples
--------

The below jupyter notebooks give examples of using LeakageLib for polarimetric fitting.

.. toctree::
    :maxdepth: 2
    :caption: Jupyter notebook examples

    examples/point-source-fit.ipynb
    examples/extended-source-fit.ipynb
    examples/time-varying-fit.ipynb
    examples/mcmc-fit.ipynb