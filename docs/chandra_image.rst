Generating a Chandra image
==========================

`LeakageLib` provides a command line utility to create a model image of an object observed by Chandra from the Chandra events. The events are rescaled to IXPE's effective area, and the exposure map is taken into account. These are the steps you should follow.

1. Download the relevant Chandra data and making ARFs for each data set (preferably with mkwarf). Currently, the code requires ACIS observations.
2. (Optional) If you want to exposure weight the image, you will also have to merge the observations with merge_obs.
3. Draw background and source regions for your object
4. Save these regions in IXPE coordinates. The fastest way to do these is to copy the regions from the Chandra events file to the IXPE events file using DS9 and then save the IXPE region in ciao format with physical coordiates (WARNING: IXPE physical coordinates must be used, not Chandra.)
5. Run the leakagelib_cxo script (see below)
6. Import the image into your LeakageLib fit using the :func:`leakagelib_cxo.cxo_source` function.

Example
-------

`LeakageLib`'s Chandra image generation tool is a command line utility. Read the help menu with

.. code-block:: bash

    python -m leakagelib_cxo -h

It will give instructions on how to use the tool. Here are some additional notes:

- You must provide an ARF for each event file with the cxo-arf, cxo-evt flags respectively
- The exposure map flag is optional. If not used, the Chandra observation will not be exposure corrected
- The width flag is optional. If not used, the width will be the full Chandra field.
- The reg-src field is optional. If not used, all Chandra events will be rescaled to the IXPE band
- The reg-bkg field is optional. If not used, the background will not be subtracted.

However, a full analysis should provide all these flags for best results.