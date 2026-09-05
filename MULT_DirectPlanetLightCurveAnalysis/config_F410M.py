#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for exoplanet light curve analysis.
"""


import datetime

# Get current datetime and format it for directory naming
def get_datetime_string():
    """Return a formatted datetime string suitable for directory naming."""
    now = datetime.datetime.now()
    # Format: YYYY-MM-DD_HHMMSS (e.g., 2025-05-14_153022)
    return now.strftime("%Y-%m-%d_%H%M%S")


# Data paths
DATA_ROOT = '../../Data/F410M_LIKELY_Th8'  # Base directory for data
PID = 4758
FIT_PLANET_FACTOR = False
FIT_PLANET_FACTOR_VALIDATION = True  # Set to True to fit planet factor during validation

# Output directories
LIGHTCURVE_DIR = f'{DATA_ROOT}/lightcurves'
if FIT_PLANET_FACTOR:
    PLOT_DIR = f'{DATA_ROOT}/plots/Aperture_Photometry_Results_Free_{get_datetime_string()}'
else:
    PLOT_DIR = f'{DATA_ROOT}/plots/Aperture_Photometry_Results_Fix_{get_datetime_string()}'

PLANET_CONTRIBUTION_DIR = f'{DATA_ROOT}/planet_contribution'
ObservationKey = 'JWST_NIRCAM_NRCALONG_F410M_MASKRND_MASK335R_SUB320A335R'

# Filter configuration
FILTER = 'F410M'  # Set to None to disable filter selection and load all filters
FILTER_CONFIGS = {
    'F410M': {
        'pixscale': 0.06242,
        'PA_offset': 0,
        'separation': 540.68335,
        'separation_err': 2.18896,
        'pa': 30.18906,        
        'pa_err': 0.21968,
        'flux': 424682.13285
    }
}

# Aperture settings

APERTURE_RADIUS = 2.0
COMPARISON_PA_LIST = [90, 150, 210, 270, 330, 180, 240, 300]
COMPARISON_SEPARATION_LIST = [542, 542, 542, 542, 542, 720, 720, 720]  # in mas

VALIDATION_PA_LIST = [0, 60, 120, 180, 240, 300, 0, 60, 120, 30, 90, 150, 210, 270, 330]
VALIDATION_SEPARATION_LIST = [400, 400, 400, 400, 400, 400, 650, 650, 650, 1300, 1300, 1300, 1300, 1300, 1300]  # in mas
ENABLE_VALIDATION = True  # Set to False to skip validation
SAVE_SAMPLES=100

# Star position (center of the frame)
STAR_X_CENTER = 174.5
STAR_Y_CENTER = 174.5

# MCMC settings
MCMC_WALKERS = 64
MCMC_STEPS = 2000
MCMC_BURNIN = 100

# Light curve model priors
LC_PRIORS = {
    'A': (-0.1, 0.1),       # Amplitude sin component
    'B': (-0.1, 0.1),       # Amplitude cos component
    'P': (5, 15),          # Period range
    'w0': (0.95, 1.05),    # Systematic offset
    'w1': (-10, 10),       # PC1 weight
    'w2': (-10, 10),       # PC2 weight
    'w3': (-10, 10),       # PC3 weight
    'planet_factor': (0, 1.0)  # Planet factor for flux adjustment
}

# Fourier model priors
FOURIER_PRIORS = {
    'A': (-1, 1),          # First sin amplitude
    'B': (-1, 1),          # First cos amplitude
    'P': (0.1, 20),        # First period
    'A2': (-1, 1),         # Second sin amplitude
    'B2': (-1, 1),         # Second cos amplitude
    'P2': (0.1, 20),       # Second period
}

# PCA settings
N_COMPONENTS = 7
FILTER_SIGMA = 1  # Gaussian filter sigma for systematic model calculation

# Parallel processing
USE_PARALLEL = True