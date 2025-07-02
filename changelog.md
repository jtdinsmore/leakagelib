# Version 2

## Potentially breaking changes
* Changed all instances of `detector_index` (0, 1, or 2) to `detector` (1, 2, or 3)
* Reversed order of `source` and `prepath` in `IXPEData.load_all_detectors_with_path` to maintain similarity with `IXPEData.__init__`
* Stopped automatic centroiding of the image. If you wish to centroid the image, there are two new functions to do so named `iterative_centroid_center` or `centroid_center`. You may also center manually `explicit_center`.
* Removed `time_cut_frac` as an argument to initializing `IXPEData`. If you wish to cut based on time, use the more flexible `cut` method.

## Features
* PSF weighted fitting
* Background de-weighting
* More documentation
* `IXPEData` changes:
    * Allowed cutting events by region
    * Allowed generation of leakage maps without using mu
    * Print statements for debugging which files are being loaded
    * Exposure map loading
* `Source` changes:
    * Added a method for loading uniform source objects
* Other small changes

## Bug fixes
* None
