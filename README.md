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
* Water demands and wet/dry conditions are determined based on historical median release, storage, and inflow values. These are also computed by `data_cdec.py` and saved in `historical_medians.csv`. 
* The scenario data can be downloaded [here](https://www.dropbox.com/s/gmgujninm02l0e8/scenario_data.zip?dl=1). Unzip and move the folders into `data/cmip5` and `data/lulc`. 
  - The CMIP5 climate scenarios are from [USBR](https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html#About) and contain daily reservoir inflows in cfs. 
  - The LULC scenarios are from multiple models and have been converted to water demand multipliers as described in [Cohen et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021WR030433)

#### Parameter training (optional): `train_historical.py`
* Fits values for reservoir policies, gains, and delta pumping. Saved in `data/params.json`.

#### Run simulations: `simulate_historical.py` or `simulate_future.py`
* Saves annual objectives (water supply and flood control metrics) in the `output/` directory as CSVs.
* The historical run also saves daily timeseries in `output/sim_historical.csv`.

#### Plot results: `plot_results_historical.py` or `plot_results_future.py`.
* The historical results are compared to observations. The future results show the distribution of performance outcomes over the ensemble of (# CMIP5 x # LULC) scenarios.

#### License: [MIT](LICENSE.md)

3.1 Climate and land use projections: : The original input climate data files should first be obtained from the repository jscohen4/orca_cmip5_inputs and the directory input_climate_files put in the directory orca/data. To process climate data, run baseline-cc-parallel.py remotely on 97 processors or baseline-cc.py on 1 processor. To ensure that climate data is processed to be input to ORCA, ensure that calc_indices = True and climate_forecasts = True in these scripts. This will cause the sript to run orca/data/calc_indices.py and orca/data/forecasting.py. The original data for USBR CMIP5 climate and hydrology projections are also publically available.

3.1.2 Water demand and land use projections: Demand data should be added as orca/data/demand_files. The demand data is further processed by running one of the baseline simulation scripts with tree_input_files = True set. Original demand data is also publically available for USGS FORE-SCE CONUS, USGS LUCAS California, and DOE GCAM CONUS.

3.2.3 Multi-objective optimization:optimization.py performs the policy search over the 235 testing scenarios. Optimized policies and objective values are stored as pickle files in the snapshots directory. If running the script in parallel, set parallel = True on line 112. The number of processors used for this optimization must be equal to population_size (line 109 in optimization.py) multiplied by the length of sc_split (line 100). If only running on one processor, set pararallel = False.
