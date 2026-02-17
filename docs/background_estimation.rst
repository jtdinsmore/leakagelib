Background particle weighting
=============================

`LeakageLib` provides a command line utility to weight each event by a likelihood to represent a particle, which we call the "particle character". These are useful to remove background pollution, especially for faint sources.

The traditional method is to cut particles. This makes a trade-off between a overly hard cut (which removes some source photons) and an overly soft cut (which fails to cut some background events). The optimal tradeoff is different for each source, because it depends on the relative ratio of photons to particles in the image.

`LeakageLib`'s particle characters help because they allow events whose particle nature is uncertain to be de-weighted instead of fully cut. This reduces the number of source photons cut, and improves uncertainties. Furthermore, it is easy to find the optimal way to treat particles using particle characters. By including the flux of the particle source as a free parameter, the best-fit model will already optimally de-weight particles. `LeakageLib`'s polarimetric fit package does this.

Example
-------

`LeakageLib`'s background weighting tool is a command line utility. Run it with

.. code-block:: bash

    python -m leakagelib_bkg L2_FILE L1_FILE OUT_FILE

This command can be run from any directory, once you have installed leakagelib. ``L2_FILE`` should be the path of the level 2 event file and ``L1_FILE`` should be the path of the level 1 event file. The number of events in each file need not agree; `leakagelib_bkg` will use event times to identify events. The output file will be stored in ``OUT_FILE``, and will have a column `BG_PROB` which contains the particle characters.