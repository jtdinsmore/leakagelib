# Version 2

## Potentially breaking changes
* Changed all instances of `detector_index` (0, 1, or 2) to `detector` (1, 2, or 3)
* Reversed order of `source` and `prepath` in `IXPEData.load_all_detectors_with_path` to maintain similarity with `IXPEData.__init__`
* Stopped automatic centroiding of the image. If you wish to centroid the image, there are two new functions to do so named `iterative_centroid_center` or `centroid_center`.

## Features
* Point source Q, U fitting
* Background removal
* Allowed cutting events by region
* Allowed generation of leakage maps without using mu
* Print statements for debugging which files are being loaded
* More documentation
* Created a new function to recenter the data set to a specific position
* Created a new function to load the exposure map