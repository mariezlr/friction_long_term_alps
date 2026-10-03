# friction_long_term_alps

Code and processed data to reproduce the analyses and figures of:

> Zeller, M., Gilbert, A., and Gimbert, F.: Constraining the glacier basal friction law from multidecadal- to century-scale observations of surface velocity and thickness changes on Alpine glaciers, *under reviw in JGR*, doi:10.XXXX/XXXXX, YYYY.

## Contents

```
data/
  obs_raw/                  In-situ observations (surface elevation and velocity) at each stake
  structural/               Glacier geometry: outlines, flowlines, bedrock DEMs, slopes
  processed_timeseries/     Processed timeseries and friction law fits
  elmer_raw/mw{1,3,6}/      Elmer/Ice outputs at DEM dates, for m = 1, 3 and 6 (m = 3 only on GitHub; all on Zenodo)
  uncertainties/            Elmer/Ice outputs of the sensitivity experiments at Argentière Profile 4 (Zenodo only)
src/
  utils.py                  Glacier and stake configuration (GLACIERS), paths and helper functions
  friction_laws.py          Friction laws and fitting functions
  slope_calculation.py      Mean surface slope at each stake
  generate_As_variable.py   Spatially variable As fields for the sensitivity experiments
  process_timeseries.py     Basal shear stress and sliding velocity timeseries
  process_uncertainties.py  Same, for the sensitivity experiments
  run_friction_fits.py      Friction law fits
  plots/                    Figures of the manuscript and supplementary material
figures/                    Figures (Fig_*.pdf)
```

The GitHub repository contains the code and the processed data. The Elmer/Ice outputs (`data/elmer_raw/`, `data/uncertainties/`) and the surface DEMs (`data/structural/surfaces/`) are too large for GitHub and are only available in the Zenodo archive (doi:10.5281/zenodo.XXXXXXX).

Processed outputs are stored in `processed_timeseries/mw{1/m}/`, where m is the exponent of the friction law used in Elmer/Ice (`mw1.000`, `mw0.333` and `mw0.167` for m = 1, 3 and 6).

## Raw data sources

The Elmer/Ice finite-element software is open source and available at https://github.com/ElmerCSC/elmerfem (Gagliardini et al., 2013).

Observations used in this study come from:
- Swiss glaciers: GLAMOS (https://www.glamos.ch); Bauder (2016) and Bauder et al. (2022) for thickness change and surface velocity; Bauder et al. (2007) for surface DEMs; Grab et al. (2021) for bedrock DEMs.
- French glaciers: GLACIOCLIM (https://glacioclim.osug.fr); Vincent et al. (2000) for Saint-Sorlin and Vincent et al. (2009) for Argentière.

## Installation

With conda:
```
conda env create -f environment.yml
conda activate friction_alps
```

Or with pip:
```
pip install -r requirements.txt
```

## Reproducing the results

The Elmer/Ice simulations are not rerun; their outputs are provided in the Zenodo archive. The mean slopes (`data/structural/slopes/`) and the spatially variable As fields used in the simulations are also provided. From the repository root:

```
python src/process_timeseries.py
python src/process_uncertainties.py
python src/run_friction_fits.py
python src/plots/main_plots.py
```

1. `process_timeseries.py` builds the basal shear stress and sliding velocity timeseries at each stake, for m = 1, 3 and 6.
2. `process_uncertainties.py` does the same for the sensitivity experiments at Argentière Profile 4.
3. `run_friction_fits.py` fits the friction law at each stake.
4. `main_plots.py` produces the main figures. The friction law fits of the sensitivity experiments are computed at the first call and saved in `processed_timeseries/uncertainty_fits/`. The other scripts in `src/plots/` produce the supplementary figures.

The GitHub repository includes the Elmer/Ice outputs for m = 3 only; the outputs for m = 1 and m = 6 are available in the Zenodo archive.

## Funding

This work was funded by the ERC project REASSESS led by Florent Gimbert (grant agreement 101126009).
