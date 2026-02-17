.. LeakageLib documentation master file, created by
   sphinx-quickstart on Fri Feb 13 14:55:39 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LeakageLib documentation
========================

`LeakageLib` provides tools for polarimetric analysis with IXPE data. The three primary tools are

1. A fitting library which uses advanced weights to deliver precise polarization estimates
2. A neural net which assigns each event a likelihood of representing a particle.
3. Software to predict polarization leakage patters

For details and examples, please see the "Polarization fitting," "Background particle weighting," and "Leakage Prediction" pages. This site also provides full API documentation of `LeakageLib`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   fitting
   background_estimation
   leakage_prediction
   faq
   leakagelib
   leakagelib_bkg