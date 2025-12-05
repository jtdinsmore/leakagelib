# LeakageLib examples

This directory contains examples for the LeakageLib's two main functions: polarization leakage prediction (`leakage-prediction` directory) and precise fitting using various weights (`polarization-extraction` directory).

* `leakage-prediction`:
    * `point-source.py` (**Start here**) Predict the polarization leakage of a point source
    * `bkg-fit.py` Fit background parameters to a background region. This step is not necessary in most analyses, since it's better to simultaneously fit background and source parameters to the entire data set. But it can be helpful sometimes to fit for background parameters first, to data within a background region.
    * `predict.py` Predict the polarization leakage of an extended source
    * `extract.py` Deconvolve the polarization leakage pattern
    * `leakage-severity.py` Generate a plot of the polarization leakage PD distribution for NN and Mom reconstruction methods
* `polarization-extraction`:
    * `point-source-fit.py` (**Start here**) Extract the polarization of a point source with background
    * `extended-source-fit.py` Extract the polarization of an extended source
    * `time-varying-fit.py` Demonstrates some ways to handle time varying sources.