`leakagelib_cxo` API documentation
==================================

:code:`leakagelib_cxo` is a command-line utility provided with LeakageLib. It creates a spatial weight template for use in LeakageLib polarimetric fitting, using Chandra data. :code:`leakagelib_cxo` produces an image which can be loaded into LeakageLib with 

.. code-block:: python

   source = leakagelib.Source.load_file(PATH_TO_LEAKAGELIB_CXO_OUTPUT)

Then the :code:`source` object can be used as described in the polarization fitting documentation.

The code works by selecting all Chandra events within the provided region, rescaling them to match IXPE's band, and subtracting background. The result is a prediction of the surface brightness map in IXPE's band. A slight blur is applied to prevent negative fluxes.

Note that you do not have to make spatial weight maps with leakagelib_cxo. You can also make them yourself and save them as FITS files.

Parameters
----------

Broadly speaking, :code:`leakagelib_cxo` requires Chandra event files, and ARFs for both IXPE and Chandra. It also requires the Chandra region within which you want to a spatial weight map. Optionally, you can provide a Chandra exposure map if you wish to exposure correct the Chandra data, and a background region file if you wish to subtract background flux. You can also change basic parameters, such as the energy range you are using and the desired width of the image. For a detailed list of the parameters, run :code:`python -m leakagelib_cxo -h`.

Notes
-----

**Format of the region files** The region files provided to leakagelib_cxo should be in ciao format, sky coordinates, stored as sexagesimal. DS9 does this by default if you use format: CIAO and coordinate system: WCS.

**Treatment of fluxes** By default, the fluxes of LeakageLib sources are left as fit parameters. The absolute scale of these fluxes are arbitrary, but the ratio of fluxes between two components do carry physical meaning. The ratio between two sources' fluxes in LeakageLib should equal the ratio of the summed :code:`leakagelib_cxo` maps. In other words, consider a fit with two extended sources S1 and S2, and each has their own :code:`leakagelib_cxo` map M1 and M2. Then sum(M1) / sum(M2) should equal flux(S1) / flux(S2), where sum(M) refers to the summed pixel values of map M, and flux(S) refers to the leakagelib flux value of source S. You may find that the best-fit fluxes do not exactly satisfy this relation. This sometimes occurs due to minor imperfections in the PSF model. The recommended approach is to manually fix flux(S1) and flux(S2) such that the ratios are correct.

**Meaning of the image** The output of :code:`leakagelib_cxo` is intended for internal use by LeakageLib, and we encourage the user not to use it for other purposes. Instead, the user should use standard ciao tools to treat the Chandra data.