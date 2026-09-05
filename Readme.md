# Conda and JWST environment:

environment.yml

CRDS_CONTEXT=jwst_1371.pmap

# Code structures

## Step 1: process uncal data

Set parameters in `FXXX_config.py` and run reduction using `calibrationPipeline.py`. The configuration file is selected at the top of the `calibrationPipeline.py` by

`import FXXX_config as config`

parameters to modify in

```
# General configuration
DATA_ROOT = '../../Data/F210M_LIKELY_Th8'  # Base directory for data
FILTER = 'F210M'             # Filter to process
PID = 4758                   # Program ID
PLOTS_DIR = 'plots'          # Directory to store plots
ObservationKey = 'JWST_NIRCAM_NRCA2_F210M_MASKRND_MASK335R_SUB320A335R'

# Recentering refernce file name. The first file in the sequence should be used by default.
# I set it explicitly to avoid confusion
RECENTER_REF = 'jw04758001001_03106_00001_nrca2_calints.fits'

# Target star information, for webbpsf generation
TARGET_SPECTRAL_TYPE = 'A6V'
```

File structure after this step should be similar to this:


## Step 2: primary subtraction and pixel level flux contribution calculation

Several pyklip scripts are provided. They should be modified accordingly to match your directory structures and filters

- `pyKLIP_all.py` Perform pyKLIP using the entire sequence
- `pyKLIP_chunk.py` Perform pyKLIP on subset of data
- `FXXX_pyKLIP_fm_all.py` Perform pyKLIP forward modeling using the entire sequence

PSF model for forward modeling is generated using `createWebbPSF.py`

Planet's flux contribution at pixel levels are derived using `FXXX_calcPlanetContribution.py`

After Step 2, the file structure should look like this:

![1763239288875](image/Readme/1763239288875.png)

## Step 3: calculate host star light curve model

- Extracting light curves: `FXXX_directStarLC_extractor.ipynb`
- Fit the stellar model: `Fit_stellarLightCurveModel.py`

Note that code for this step is ad hoc. It works well for beta Pic. It should be carefully tested for HR8799. Other methods may work better.

## Step 4: analyze planet light curve

- run the `directPlanetLightCurveAnalyzer.py` script
- Configuration method is similar to Step 1.

## Step 5: injection and recovery validation

- See [`readme_injection_recovery.md`](readme_injection_recovery.md) for the standalone injection/recovery workflow.

# Notes for Beta Pic results

Several slides decks for Beta Pic results are in the Notes directory. It has useful information for reducing the HR 8799 data
