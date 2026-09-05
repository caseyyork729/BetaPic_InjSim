# Beta Pic b injection and recovery handoff

This directory contains the code needed to inject a time-variable PSF into the processed NIRCam time-series data and recover the injected light curve. It is intended for a grid of injections in F210M and F410M. The workflow does not require re-running JWST calibration, pyKLIP, or forward modeling.

## What is included

The two executable stages are:

1. `InjectionRecovery/psfInjectionSimulator.py` injects a PSF into the processed `calints.fits` cubes.
2. `InjectionRecovery/recoverInjectedLightCurve.py` performs aperture photometry, PCA-based systematic correction, light-curve fitting, and a combined Fourier fit.

The recovery stage imports the modules in `DirectPlanetLightCurveAnalysis/`. `misc.py` is a local dependency that was missing from the original handoff; it is now included here. The remaining dependencies are packages from `environment.yml`, especially `spaceKLIP`, `astropy`, `scipy`, `photutils`, `h5py`, `emcee`, `corner`, `scikit-learn`, and `joblib`.

## Required data layout

Place the data relative to this README as follows. The FITS files and PSF files are intentionally not included in the code handoff.

```text
BetaPic/
├── Data/
│   ├── F210M_LIKELY_Th8/
│   │   └── aligned/
│   │       └── *calints.fits
│   ├── F410M_LIKELY_Th8/
│   │   └── aligned/
│   │       └── *calints.fits
│   ├── F210M/
│   │   └── psf_models_blurred/
│   │       └── webbpsf_b_pic_b_F210M_0.fits
│   └── F410M/
│       └── psf_models_blurred/
│           └── webbpsf_b_pic_b_F410M_0.fits
├── DirectPlanetLightCurveAnalysis/
└── InjectionRecovery/
```

The input directory must contain the processed, aligned science `calints.fits` files. The current recovery code assumes the observation contains 20 integrations in database order: the first 10 are roll 0 and the next 10 are roll 1. Each FITS file must contain the extensions used by the existing reductions:

- extension 1: science image cube;
- extension 2: uncertainty cube;
- extension 5: `int_mid_BJD_TDB` integration times;
- extension 10: two-column x/y centering shifts.

The observation key, star center, pixel scale, Beta Pic b position, and PSF filename are filter-specific in `InjectionRecovery/psfInjectionConfig_F210M.py` and `psfInjectionConfig_F410M.py`. Update those values if the processed data have different metadata or filenames.

## Environment

Use the supplied `JWST_HCI` environment. If it is already available, do not recreate it. Otherwise, create it from `environment.yml` using the normal local conda workflow. The code paths are resolved from the script location, so commands can be run from the handoff root:

```bash
cd /path/to/BetaPic
export CRDS_CONTEXT=jwst_1371.pmap

# Import/syntax check only
conda run -n JWST_HCI python -m py_compile \
  InjectionRecovery/psfInjectionSimulator.py \
  InjectionRecovery/recoverInjectedLightCurve.py \
  InjectionRecovery/psfInjectionUtils.py \
  DirectPlanetLightCurveAnalysis/*.py
```

Before a full run, set `USE_PARALLEL = False` in the selected injection config and use a small copied input directory if you want a cheap file-level test. The full recovery stage includes MCMC fits and is not a smoke test.

## Configure and run one simulation

Select the filter through `JWST_FILTER`; no source edit is needed to switch between F210M and F410M:

```bash
cd /path/to/BetaPic

JWST_FILTER=F210M conda run -n JWST_HCI \
  python InjectionRecovery/psfInjectionSimulator.py

JWST_FILTER=F210M conda run -n JWST_HCI \
  python InjectionRecovery/recoverInjectedLightCurve.py
```

Use `JWST_FILTER=F410M` for the F410M configuration. Edit the corresponding `psfInjectionConfig_*.py` file before each run to change, for example, separation, position angle, base flux, fractional amplitude, period, phase, and pointing-error multiplier. The default output directory label includes these quantities, including period and phase, so grid points do not silently overwrite one another. If two runs have the same parameter values but differ in code or analysis settings, change `RUN_LABEL` manually to keep them separate.

Do not run the recovery stage until the injection stage has completed successfully and produced both the simulated `aligned/` files and the `planet_contribution/` products. The simulator removes the nominal Beta Pic b PSF contribution before adding the test injection; this is part of the current experiment definition.

## Output products

For each configuration, products are written below:

```text
Data/<FILTER>_LIKELY_Th8/simulated_data/<RUN_LABEL>/
├── aligned/                         # injected calints FITS files
├── planet_contribution/             # injected PSF and pixel-level diagnostics
├── lightcurves/
│   ├── Planet_LightCurves_roll_*.hdf5
│   ├── Planet_LightCurve_Corrected_Roll_*.txt
│   ├── recovery_summary.json
│   └── recovery_posterior.npz
└── plots/Aperture_Photometry_Results/
    ├── apertures/
    ├── lightcurves/
    ├── pca/
    ├── modeling/
    └── validation/                  # only if ENABLE_VALIDATION=True
```

`recovery_summary.json` is the primary grid-analysis product. It records the injected parameters, the least-squares and MCMC Fourier results, recovered amplitude, recovered period, recovered phase, and MCMC settings. `recovery_posterior.npz` contains the posterior samples and fit vectors for more detailed analysis.

## Post-simulation analysis

For each run, first inspect:

1. `planet_contribution/` to confirm the injected PSF is at the requested separation and PA in both rolls.
2. The raw-light-curve and PCA plots to check aperture placement, comparison-aperture behavior, and the number of useful principal components.
3. The per-roll modeling plots and `*_combined_fourier_fit.png` to compare the recovered waveform with the injected waveform.
4. `lightcurves/recovery_summary.json` for machine-readable values across a grid.

The Fourier model is

```text
1 + A sin(2 pi t / P) + B cos(2 pi t / P)
```

The summary reports `amplitude_fraction = sqrt(A^2 + B^2)` and `phase_rad = atan2(B, A)`. Compare the recovered `period_hours` and `amplitude_fraction` to the values in the `injected` block. Use `recovery_posterior.npz` when uncertainties or posterior correlations are needed.

For a broad search, collect one `recovery_summary.json` from each run into a table keyed by filter and `simulation_root`. At minimum, retain separation, PA, injected base flux, injected amplitude, injected period, pointing-error multiplier, recovered amplitude, recovered period, and any fit-quality or convergence flags you add to the grid analysis. Treat a completed JSON file as a completed analysis product, not automatically as a converged fit: inspect the posterior and the modeling plots for boundary-hitting or poorly mixed chains.

## Important assumptions and limitations

- The code currently assumes two rolls with 10 files per roll. Change the roll slicing in `recoverInjectedLightCurve.py` if the processed sequence has a different layout.
- `psfInjectionSimulator.py` creates a `spaceKLIP` database in the simulation data root. Do not reuse an output directory for a different parameter set.
- The recovery fit uses the Beta Pic b stellar light-curve model and the comparison-aperture PCA correction already present in this handoff.
- `ENABLE_VALIDATION` is `False` by default because validation adds many MCMC fits. Turn it on only for selected runs.
- Keep F210M and F410M configuration edits separate. Do not mix a filter's PSF, pixel scale, observation key, star center, or Beta Pic b flux with the other filter.
