# Exoplanet Light Curve Analysis

This package provides tools for analyzing time-series images of exoplanets, extracting light curves, removing systematic noise, and identifying signals.

## Overview

The script processes time-series images of an exoplanet to:

1. Extract light curves from aperture photometry
2. Remove systematic noise using PCA
3. Model the planet signal and identify variations
4. Visualize the results

## File Structure

- `exoplanet_analysis.py`: Main script
- `config.py`: Configuration settings
- `utils.py`: Utility functions
- `aperture_photometry.py`: Functions for aperture photometry
- `modeling.py`: Light curve modeling and systematic removal
- `visualization.py`: Plotting functions
- `BetaPicStellarLCModel.py`: Stellar light curve model (external)
- `perform_aperture_photometry.py`: Aperture photometry implementation (external)

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Astropy
- h5py
- emcee
- corner
- scikit-learn
- joblib
- spaceKLIP

## Configuration

All settings are defined in `config.py`, including:

- Data paths
- Filter settings
- Aperture parameters
- MCMC settings
- Model priors

## Usage

1. Set the desired parameters in `config.py`
2. Run the main script:

```bash
python exoplanet_analysis.py
```

## Output

The script generates:

1. HDF5 files with extracted light curves
2. Text files with corrected planet light curves
3. Plots for aperture positions, raw light curves, PCA components, and model fits

All outputs are saved to the configured directories.

## Method

The analysis follows these steps:

1. Calculate planet positions using separation and position angle measurements
2. Define comparison aperture positions for systematic noise characterization
3. Extract light curves from all apertures
4. Calculate the planet contribution in each aperture
5. Model the stellar light curve
6. Identify systematic noise patterns using PCA
7. Fit a combined model (stellar + planet + systematics)
8. Extract the planet signal
9. Combine data from both telescope rolls
10. Fit a Fourier model to the combined light curve

## Directory Structure

```
DATA_ROOT/
├── aligned/             # Input FITS files
├── lightcurves/         # Output light curve data
├── planet_contribution/ # Planet PSF models
└── plots/               # Output plots
    ├── apertures/       # Aperture position plots
    ├── lightcurves/     # Raw light curve plots
    ├── pca/             # PCA component plots
    └── modeling/        # Model fit plots
```
