Leakage Prediction
==================

Polarization leakage is a spurious polarization pattern that occurs in IXPE data, even when imaging an unpolarized point source. The maximum PD of polarization leakage patterns can be quite high, and the patterns extend for up to an arcminute.

`Bucciantini (2023) <https://ui.adsabs.harvard.edu/abs/2023A%26A...672A..66B/abstract>`_ first explained polarization leakage by assuming that errors in the reconstructed position of each event are correlated with the event EVPA. `LeakageLib` builds on that modeling by measuring IXPE's point-spread functions for each detector from on-sky data, and using them to modify and extend the polarization leakage model. Polarization leakage patterns were matched to quite good precision.

Examples
--------

See below for several jupyter notebooks containing increasingly advanced examples of using LeakageLib for leakage prediction.

.. toctree::
    :maxdepth: 2
    :caption: Jupyter notebook examples

    examples/leakage-predict.ipynb
    examples/leakage-severity.ipynb
    examples/leakage-extract.ipynb