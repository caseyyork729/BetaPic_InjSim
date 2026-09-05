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
DATA_ROOT = '../../Data/F210M_LIKELY_Th8'  # Base directory for data
PID = 4758



# Output directories
LIGHTCURVE_DIR = f'{DATA_ROOT}/lightcurves'
PLOT_DIR = f'{DATA_ROOT}/plots/Aperture_Photometry_Results_testing330_{get_datetime_string()}'
PLANET_CONTRIBUTION_DIR = f'{DATA_ROOT}/planet_contribution'
ObservationKey = 'JWST_NIRCAM_NRCA2_F210M_MASKRND_MASK335R_SUB320A335R'

# Filter configuration
FILTER = 'F210M'  # Set to None to disable filter selection and load all filters
FILTER_CONFIGS = {
    'F210M': {
        'pixscale': 0.03068,
        'PA_offset': 0,
        'separation': 541.96546,
        'separation_err': 0.66366,
        'pa': 330,        
        'pa_err': 0.07349,
        'flux': 1455230.39320,
    }
}

# Aperture settings
APERTURE_RADIUS = 3.5
COMPARISON_PA_LIST = [0, 60, 90, 150, 180, 210, 240, 270, 300]
COMPARISON_SEPARATION_LIST = [542] * len(COMPARISON_PA_LIST)  # in mas
VALIDATION_PA_LIST = [0, 60, 120, 180, 240, 300, 0, 60, 120, 30, 90, 150, 210, 270, 330]
VALIDATION_SEPARATION_LIST = [400, 400, 400, 400, 400, 400, 650, 650, 650, 1300, 1300, 1300, 1300, 1300, 1300]  # in mas
ENABLE_VALIDATION = False  # Set to False to skip validation

# Star position (center of the frame)
STAR_X_CENTER = 159.5
STAR_Y_CENTER = 159.5

# MCMC settings
MCMC_WALKERS = 64
MCMC_STEPS = 2000
MCMC_BURNIN = 500

# Light curve model priors
LC_PRIORS = {
    'A': (-0.2, 0.2),       # Amplitude sin component
    'B': (-0.2, 0.2),       # Amplitude cos component
    'P': (3, 20),          # Period range
    'w0': (0.95, 1.05),    # Systematic offset
    'w1': (-10, 10),       # PC1 weight
    'w2': (-10, 10),       # PC2 weight
    'w3': (-10, 10),       # PC3 weight
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
N_COMPONENTS = 6
FILTER_SIGMA = 0.5  # Gaussian filter sigma for systematic model calculation

# Parallel processing
USE_PARALLEL = True