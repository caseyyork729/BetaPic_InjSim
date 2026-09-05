#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for PSF injection simulation.
"""

import datetime
import os
from pathlib import Path
import numpy as np

# Get current datetime and format it for directory naming
def get_datetime_string():
    """Return a formatted datetime string suitable for directory naming."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H%M%S")

# Filepaths
PSF_FILENAME = 'webbpsf_b_pic_b_F410M_0.fits'
HANDOFF_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = str(HANDOFF_ROOT / 'Data' / 'F410M_LIKELY_Th8')
SIMDATA_ROOT = os.path.join(DATA_ROOT, 'simulated_data')
DATA_TYPE = 'aligned'
INPUT_DATA_DIR = f'{DATA_ROOT}/{DATA_TYPE}'  # Directory containing input calints files
PSF_LIBRARY_DIR = str(HANDOFF_ROOT / 'Data' / 'F410M' / 'psf_models_blurred')  # Directory containing PSF files
FILE_PATTERN = '*calints.fits'  # Pattern to match input files

# Image info
TARGET_PIXEL_SCALE = 0.06242  # arcsec/pixel
PID = 4758
OBSERVATION_KEY = 'JWST_NIRCAM_NRCALONG_F410M_MASKRND_MASK335R_SUB320A335R'
t0=60755.475234183694

#############################
###INJECTION CONFIGURATION###
#############################

TARGET_SEPARATION = [400, 540, 800, 1080, 1500]
TARGET_PA = [90, 180, 270]

FLUX_BASE = [42000, 100000, 200000, 300000, 420000]
FLUX_AMPLITUDE = 0.01
FLUX_PERIOD = 8.5  # Period in hours
FLUX_PHASE = np.random.default_rng(FLUX_BASE[0]*TARGET_PA).uniform(-np.pi, np.pi)

DXY = 0
DXY_MULTIPLIER = 0

# Filter configuration
FILTER = 'F410M'  # Set to None to disable filter selection and load all filters
FILTER_CONFIGS = {
    'F410M': {
        'pixscale': TARGET_PIXEL_SCALE,
        'PA_offset': 0,
        'separation': TARGET_SEPARATION,
        'separation_err': 0,
        'pa': TARGET_PA,        
        'pa_err': 0,
        'flux': FLUX_BASE
    }
}

# Aperture settings
APERTURE_RADIUS = 2.0
N_COMPARISON = 7

VALIDATION_PA_LIST = [0, 60, 120, 180, 240, 300, 0, 60, 120, 30, 90, 150, 210, 270, 330]
VALIDATION_SEPARATION_LIST = [400, 400, 400, 400, 400, 400, 650, 650, 650, 1300, 1300, 1300, 1300, 1300, 1300]  # in mas

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

# Star info 
STAR_X_CENTER = 174.5
STAR_Y_CENTER = 174.5
PA_BETA_PIC_B = 30.18906  # Position angle of Beta Pic b in degrees
SEP_BETA_PIC_B = 540.68335  # Separation of Beta Pic b in mas
FLUX_BETA_PIC_B = 424682.13285  # Flux of Beta Pic b in relative units

# Processing settings
USE_PARALLEL = True
VERBOSE = False
DELETE_SUPERSEDED = True
USE_MCMC = False
ENABLE_VALIDATION = False  # Set to False to skip validation
FIT_PLANET_FACTOR_VALIDATION = True