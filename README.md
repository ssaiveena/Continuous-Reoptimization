## Continuous-Reoptimization
This repository contains all code corresponding to methods and figure generation in the paper below:

Dynamic re-optimization of reservoir policies as an adaptation to climate change

#### Requirements
[NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Scipy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/stable/), [Seaborn](https://seaborn.pydata.org/), [Numba](https://numba.pydata.org/),  [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

### Directories

``data``: Contains input data for analysis used in the study. This data is read from the file ``Reopt_main.py``

``reopt``: This folder contains function codes called by ``Reopt_main.py``. This directory also has ``Regression_model.py`` that investigates a regression model to predict how the policy parameters would change over time based on feature variables describing changes in hydrology and demand.

``figures``: Directory containing python scripts to generate Figures 3-5 of the manuscript.

### Data preparation and model run
* Network components are defined in `data/nodes.json`
* Historical data can be updated to the current day from [CDEC](https://cdec.water.ca.gov/). Run with `updating=True`.
* Water demands and wet/dry conditions are determined based on historical median release, storage, and inflow values. These are also computed by `data/data_cdec.py` and saved in `historical_medians.csv`. 
* The scenario data can be downloaded [here](https://www.dropbox.com/s/gmgujninm02l0e8/scenario_data.zip?dl=1). Unzip and move the folders into `data/cmip5` and `data/lulc`. 
  - The CMIP5 climate scenarios are from [USBR](https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html#About) and contain daily reservoir inflows in cfs. 
  - The LULC scenarios are from multiple models and have been converted to water demand multipliers as described in [Cohen et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021WR030433)

#### Parameter training (optional): `train_historical.py`
* Fits values for reservoir policies, gains, and delta pumping. Saved in `data/params.json`.
* These reservoir policies are considered as simulated or baseline in the manuscript

### Re-optimization: 
* Reopt_main.py performs the policy search over the cmpi5 and lulc scenarios. The parameters of the rule curve are adapted by optimizing the reservoir performance based on the recently observed historical window (w) years of data and evaluating the performance of this policy during the following frequency of operation (f) years before the next re-optimization occurs.

#### License: [MIT](LICENSE.md)
