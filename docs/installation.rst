.. LeakageLib documentation master file, created by
   sphinx-quickstart on Fri Feb 13 14:55:39 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installing LeakageLib
========================

1. To install LeakageLib, clone this repository and install it with `pip`:

.. code-block:: bash

   git clone https://github.com/jtdinsmore/leakagelib.git
   cd leakagelib
   python -m pip install -e .

2. (optional) To make loading data more convenient, change the `DATA_DIRECTORIES` in **src/settings.py** variable to point to where you store your IXPE data files. You can list multiple directories. Alternatively, you can use the :meth:`leakagelib.ixpe_data.IXPEData.load_all_detectors_with_path` function in the script to load all your data files, and feed in the directory to the data each time.