FAQ
===

What are the use cases of LeakageLib
------------------------------------

LeakageLib is general-purpose. But because it uses spatial weights, there are a few places when it is most useful:

* Faint sources
* Extended sources
* Overlapping sources
* Fitting a polarization model that varies over time

How do I cite `LeakageLib`?
---------------------------

For the `LeakageLib` polarization fitting or background weights, cite `Dinsmore & Romani 2025 <https://ui.adsabs.harvard.edu/abs/2025ApJ...993..173D/abstract>`_
For the `LeakageLib` leakage prediction software and the PSFs, cite `Dinsmore & Romani 2022 <https://ui.adsabs.harvard.edu/abs/2024ApJ...962..183D/abstract>`_

Where can I find the source code?
---------------------------------

The source code is located on `GitHub <https://github.com/jtdinsmore/leakagelib.git>`_

Whom do I contact about bugs?
---------------------------------

Please contact Jack T. Dinsmore to report a bug, or for help using LeakageLib.

What if I have a different question?
------------------------------------

You can contact Jack T. Dinsmore via his email.

What are the units of LeakageLib fluxes?
----------------------------------------
The absolute scale of LeakageLib fluxes is arbitrary. I.e. a flux of f1=1 has no meaning. However, relative fluxes matter, meaning that if f2 is the flux of a second fit source, and f2 = 2*f1, then LeakageLib believes that source 2 has twice as many counts as source 1. Note that for extended sources, the best-fit fluxes found by LeakageLib are not perfectly accurate. See the documentation on :code:`leakagelib_cxo` for more information.

What are the pitfalls I should look out for when using LeakageLib?
------------------------------------------------------------------
Please see :ref:`this section of the fitting documentation.<caveats>` for a list of LeakageLib ceaveats.