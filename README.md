# Continuous-Reoptimization
This repository contains all code corresponding to methods and figure generation in the paper below:

Dynamic re-optimization of reservoir policies as an adaptation to climate change

## Requirements
[NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Scipy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/stable/), [Seaborn](https://seaborn.pydata.org/), numba, MPI for Python (optional).

## Directories

figure-scripts: Directory containing python scripts to generate **Figures 3-11**.

## Paper methods, results and figures

The following instrutctions correspond to subsections in Section 3 Methods, and their corrsponding subsections in Section 4 Results and discussion.

3.1 Climate and land use projections: : The original input climate data files should first be obtained from the repository jscohen4/orca_cmip5_inputs and the directory input_climate_files put in the directory orca/data. To process climate data, run baseline-cc-parallel.py remotely on 97 processors or baseline-cc.py on 1 processor. To ensure that climate data is processed to be input to ORCA, ensure that calc_indices = True and climate_forecasts = True in these scripts. This will cause the sript to run orca/data/calc_indices.py and orca/data/forecasting.py. The original data for USBR CMIP5 climate and hydrology projections are also publically available.

3.1.2 Water demand and land use projections: Demand data should be added as orca/data/demand_files. The demand data is further processed by running one of the baseline simulation scripts with tree_input_files = True set. Original demand data is also publically available for USGS FORE-SCE CONUS, USGS LUCAS California, and DOE GCAM CONUS.

3.2.3 Multi-objective optimization:optimization.py performs the policy search over the 235 testing scenarios. Optimized policies and objective values are stored as pickle files in the snapshots directory. If running the script in parallel, set parallel = True on line 112. The number of processors used for this optimization must be equal to population_size (line 109 in optimization.py) multiplied by the length of sc_split (line 100). If only running on one processor, set pararallel = False.
