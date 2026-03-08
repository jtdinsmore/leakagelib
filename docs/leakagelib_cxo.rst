`leakagelib_cxo` API documentation
==================================

This code rescales Chandra event data to the IXPE effective area, so that it can be used to fit for source polarizations. Please see :doc:`chandra_image` for instructions on the command line interface for this code. This documentation describes the python API for loading it into your code.

Once you have created the rescaled image, it is very simple to import it into LeakageLib. To create a source object suitable for loading into your fitter, use :func:`leakagelib_cxo.cxo_source` to load the image you created. It will return a :class:`leakagelib.Source` object, which you can load in the usual way (see the :doc:`examples/extended-source-fit` example)

leakagelib\_cxo module
-------------------------

.. automodule:: leakagelib_cxo
   :members:
   :show-inheritance:
   :undoc-members:
