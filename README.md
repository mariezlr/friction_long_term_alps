# friction_long_term_alps

Code and processed data to reproduce the analyses and figures of the study "Constraining the glacier basal friction law from multidecadal- to century-scale observations of surface velocity and thickness changes on Alpine glaciers" (M. Zeller, A. Gilbert, F. Gimbert).

WARNING: Data and code are still being organized; final version will be released upon acceptance.


## Raw data availability

The raw observational datasets are publicly accessible from their respective repositories:

The Elmer/Ice finite-element software used in this study is open source and available at https://github.com/ElmerCSC/elmerfem ([Gagliardini et al., 2013](https://doi.org/10.5194/gmd-6-1299-2013)). 

Bedrock and surface DEMs, velocity and elevation datasets for the Alpine glaciers analyzed in this study are available from the following sources: 
- GLAMOS database (https://www.glamos.ch); 
- Swiss glaciers monitoring programs ([Bauder, 2016](https://doi.glamos.ch/pubs/glrep/glrep_133-134.html); [Bauder et al., 2022](https://doi.glamos.ch/pubs/glrep/glrep_141-142.html) for the thickness change and surface velocity timeseries; [Bauder et al., 2007](https://doi.org/10.3189/172756407782871701) for the surface DEMs; [Grab et al., 2021](http://dx.doi.org/10.1017/jog.2021.55) for the bedrock DEMs); 
- GLACIOCLIM database (https://glacioclim.osug.fr);
- French glaciers monitoring programs (Saint-Sorlin: [Vincent et al., 2000](https://doi.org/10.3189/172756500781833052); Argentière: [Vincent et al., 2009](https://doi.org/10.3189/172756409787769500))


## Project structure

```
friction_long_term_alps/
├── data/
│   ├── elmer_raw/               # Elmer/Ice outputs per stake (subdirs mw1, mw3, mw6)
│   ├── obs_raw/                 # Raw observational data (velocity, altitude, thickness)
│   ├── processed_timeseries/    # Final timeseries and friction reconstructions per stake
│   │   └── mw{value}/
│   │       └── friction_fits/
│   ├── structural/
│   │   ├── outlines/            # Glacier outlines
│   │   ├── bedrocks/            # Bedrock DEMs
│   │   ├── flowlines/           # Flowline coordinates
│   │   ├── slopes/              # Pre-computed slope CSVs (output of slope_calculation.py)
│   │   └── surfaces/            # Surface DEMs per glacier and year (.dat files)
│   └── uncertainties/           # Elmer outputs for uncertainty ensemble (varying A, C)
├── src/
│   ├── utils.py                 # GLACIERS dict: all metadata, coordinates, parameters
│   ├── slope_calculation.py     # DEM-based slope computation per stake
│   ├── process_timeseries.py    # Main pipeline: calibration + friction reconstruction
│   ├── process_uncertainties.py # Same pipeline over uncertainty ensemble
│   ├── friction_laws.py         # Friction laws, stress calculations, empirical fits
│   ├── run_friction_fits.py     # Entry point: runs fits over all glaciers/stakes
│   └── plots/                   # Scripts reproducing all manuscript figures
└── figures/                     # Output figures
```

**Glaciers:** Allalin (All), Argentière (Arg), Saint-Sorlin (StSo), Glacier Blanc (GB), Gébroulaz (Geb), Giétro (Gie), Corbassière (Cor), Mer de Glace (MDG)


## Figures

All manuscript figures can be reproduced from `src/plots/`. Output is saved to `figures/`.
