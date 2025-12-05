# friction_long_term_alps

Code and processed data to reproduce the analyses and figures of the study "Constraining the glacier basal friction law from multidecadal to century scales observations of surface velocity and thickness changes on Alpine glaciers" (M. Zeller, A. Gilbert, F. Gimbert).

WARNING: Data and code are still being organized; final version will be released upon acceptance.


## Raw data availability

The raw observational datasets are publicly accessible from their respective repositories:

The Elmer/Ice finite-element software used in this study is open source and available at https://github.com/ElmerCSC/elmerfem ([Gagliardini et al., 2013](https://doi.org/10.5194/gmd-6-1299-2013)). 

Bedrock and surface DEMs, velocity and elevation datasets for the Alpine glaciers analyzed in this study are available from the following sources: 
- GLAMOS database (https://www.glamos.ch); 
- Swiss glaciers monitoring programs ([Bauder, 2016](https://doi.glamos.ch/pubs/glrep/glrep_133-134.html); [Bauder et al., 2022](https://doi.glamos.ch/pubs/glrep/glrep_141-142.html) for the thickness change and surface velocity timeseries; [Bauder et al., 2007](https://doi.org/10.3189/172756407782871701) for the surface DEMs; [Grab et al., 2021](http://dx.doi.org/10.1017/jog.2021.55) for the bedrock DEMs); 
- GLACIOCLIM database (https://glacioclim.osug.fr);
- French glaciers monitoring programs (Saint-Sorlin: [Vincent et al., 2000](https://doi.org/10.3189/172756500781833052); Argentière: [Vincent et al., 2009](https://doi.org/10.3189/172756409787769500))



## Workflow overview

1. Data acquisition

### Geometric data
Surface and bedrock DEMs are available on Zenodo : {insert link}.

All DEMs are regridded to a common grid format to serve as input for Elmer/Ice simulations.

Lighter geometric files (glacier outlines, stake positions, etc.) are stored directly in this repository under data/raw/.

### Timeseries
Timeseries for each stake of
- raw data : surface velocity, thickness change
- processed data : deformation velocity, basal shear stress, sliding velocity
are stored in separate CSV files, one per stake under data/{glacier}_{stake}_all_data.csv.


2. Data processing and analysis

Analysis code is located in analysis/, structured as follows:

friction_laws.py -->    Functions to fit friction laws (Weertman, Lliboutry, Tsai) and compute optimized parameters.

run_friction_fits.py -->    Functions to compute the best velocity and shear stress timeseries from all processed data and to apply the best fit with the appropriate friction law for each stake.

utils.py -->    GLACIERS dictionary containing geometric data, time series and fitted parameters for each stake. Initially populated with raw data and progressively enriched with outputs from run_friction_fits.py

plotting_functions.py -->    Plot utilities that reproduce the figures of the manuscript using data stored in GLACIERS.


3. Plot figures

All important paper figures can be generated using functions in plotting_functions.py

Output figures are saved in figures/