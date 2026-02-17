Polarization fitting
====================

`LeakageLib` polarimetric fitting has two main advantages:

1. It delivers precise and accurate polarization fit results, enabled by advanced weights that help distinguish source events from background events.
2. It can fit complex fields, that contain e.g. extended sources, overlapping sources, and sources with EVPA and PD sweeps.

It was shown in `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_ that LeakageLib delivers a 2x decrease in the QU uncertainty contour area compared to the simplest polarimetric extraction method; that is, it delivers uncertainties equivalent to doubling the amount of available data.

The weights `LeakageLib` uses are

* **Spatial weights**, which help remove background and separate overlapping sources. LeakageLib's method uses weights in a way which avoids pollution from polarization leakage.
* **Polarimetric weights**, i.e. individual modulation factors for each event. Most analyses treat modulation factor as a function of event energy, and properly using these provides a very large precision gain. The neural net designed by Lawrence Peirson provides more individualized estimates of these modulation factors for each event. These provide additional gain.
* **Particle weights**, which de-weight IXPE events which are morphologically similar to particles. Most IXPE analyses cut particles based on level 1 event data, making a trade-off between cutting too few particles or too many photons. De-weighting avoids this trade-off, which increases precision
* **Phase/time weights**, if some of the sources are pulsed or time variable.
* **Energy weights**, which use event energies to distinguish between sources with different spectra. THese are useful if the spectrum is already known, but LeakageLib is not yet designed for full spectro-polarimetric fitting (i.e. simultaneously fiting spectrum parameters and polarization parameters).

These weights are incorporated using a maximum likelihood technique described in `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_.

Examples
--------

The below jupyter notebooks give examples of using LeakageLib for polarimetric fitting.

.. toctree::
    :maxdepth: 2
    :caption: Jupyter notebook examples

    examples/point-source-fit.ipynb
    examples/extended-source-fit.ipynb
    examples/time-varying-fit.ipynb