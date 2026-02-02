# Compute trends of Arctic and Arctic subregions of marine and atmospheric variables
> This project uses the aerosol-climate model ECHAM6.3-HAM2.3 output (data is archived in Levante HPC system) and ocean
> biomolecules and marine tracers from FESOM-REcoM regular interpolated grid that are stored on [Zenodo](https://doi.org/10.5281/zenodo.15172565). 
> See more information and results in [Leon-Marcos et al. (2025a)](https://doi.org/10.5194/egusphere-2025-2829) and [Leon-Marcos et al. (2025b)](https://doi.org/10.5194/gmd-18-4183-2025). 
> 
> The conda environment for this project is contained in trends.yml file. Run [start_env.py](start_env.py) to set up the environment for this project.

* Adapt required directories with models outputs and Arctic region limits and season to compute trends in [global_vars.py](utils_functions/global_vars.py).
* Adapt Arctic subregions definition, variable metadata considered for trends in [utils.py](utils_functions/utils.py) 
(the variables should be considered in alignment with those read in [main.py](compute_trends/main.py))

### Compute trends
* Run [cythonize.sh](compute_trends/cythonize.sh) to create executable for cython code ([process_statsmodels.pyx](compute_trends/process_statsmodels.pyx)), 
included to optimize trends computation. 
* Run [main.py](compute_trends/main.py) or sbatch [run_main.sh](compute_trends/run_main.sh) (on levante) to start the trend 
calculation of the variables defined in [utils.py](utils_functions/utils.py). It will create a .pkl (pickle file in python) 
with a dictionary with the trends results for each variable.
* Run [main_trend_per_ice.py](compute_trends/main_trend_per_ice.py) or uncomment line 22 and execute sbatch [run_main.sh](compute_trends/run_main.sh) 
(on levante) to start the trend calculation of emissions per unit of sea ice

#### Contributions
- Michael Weger Dr. (TROPOS) assisted in the implementation and optimization of the trend computation.

### Create maps of trends, time series and heatmap plots 
* Run [plot_trend_maps.py](trend_maps/plot_trend_maps.py) to create maps of trends over the Arctic shown in 
[Leon-Marcos et al. (2025a)](https://doi.org/10.5194/egusphere-2025-2829) and doctoral thesis at the University of Leipzig.
* Run [time_series.py](trend_time_series/time_series.py) to create time series plots as in [Leon-Marcos et al. (2025a)](https://doi.org/10.5194/egusphere-2025-2829) 
and Doctoral thesis of Anisbel Leon Marcos at Leipzig University
* Run [aerosol_heatmaps.py](trend_heatmaps/aerosol_heatmaps.py) and [biomolecule_heatmaps.py](trend_heatmaps/biomolecule_heatmaps.py) 
to create heatmap plots of trends of aerosol species and marine biomolecules as in [Leon-Marcos et al. (2025a)](https://doi.org/10.5194/egusphere-2025-2829)
and Doctoral thesis of Anisbel Leon Marcos at Leipzig University.

### Compute and plot Spearman correlation between emission and emission drivers
* Run [compute_spearman_corr.py](spearman_corr_emiss_drivers/compute_spearman_corr.py) or [run_main_corr.sh](spearman_corr_emiss_drivers/run_main_corr.sh) (on levante) to 
compute the Spearman correlation over the Arctic and for predefined Arctic subregions
* Run [compute_spearman_corr.py](spearman_corr_emiss_drivers/compute_spearman_corr.py) to plot Arctic maps of the Spearman correlation between emission and emission drivers

