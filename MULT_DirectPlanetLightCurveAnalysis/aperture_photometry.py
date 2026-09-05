#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for aperture photometry on exoplanet images.
"""

import numpy as np
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed
from astropy.io import fits
import gc

# Import from the original perform_aperture_photometry module
# Note: This import assumes the original module exists in the same directory
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.perform_aperture_photometry import perform_aperture_photometry, planet_aperture_photometry


def process_single_file(filename, xc, yc, aperture_radius):
    """
    Process a single FITS file and extract aperture photometry.
    
    Parameters
    ----------
    filename : str
        Path to the FITS file
    xc : array-like
        X coordinates of aperture centers
    yc : array-like
        Y coordinates of aperture centers
    aperture_radius : float
        Radius of aperture in pixels
        
    Returns
    -------
    dict
        Dictionary containing extracted data:
        - flux: Flux measurements
        - flux_err: Flux errors
        - time: Time measurements
        - xshift: X shifts
        - yshift: Y shifts
    """
    with fits.open(filename) as fitsfile:
        imCube = fitsfile[1].data
        errCube = fitsfile[2].data
        time_data = fitsfile[5].data
        shift_data = fitsfile[10].data
        xshift_i = shift_data[:, 0]
        yshift_i = shift_data[:, 1]

    nFrames = imCube.shape[0]
    nAper = len(xc)
    
    # Extract time data
    t_BJD_i = time_data['int_mid_BJD_TDB']

    # Pre-allocate arrays
    flux_i = np.zeros((nFrames, nAper))
    flux_err_i = np.zeros((nFrames, nAper))

    # Process each frame
    for i in range(nFrames):
        im_i = imCube[i, :, :]
        err_i = errCube[i, :, :]

        # Process each aperture
        for j in range(nAper):
            f_ij, e_ij = planet_aperture_photometry(
                im_i, 
                error=err_i, 
                x_center=xc[j], 
                y_center=yc[j], 
                aperture_radius=aperture_radius,               
            )
            flux_i[i, j] = f_ij
            flux_err_i[i, j] = e_ij

    return {
        'flux': flux_i,
        'flux_err': flux_err_i,
        'time': t_BJD_i,        
        'xshift': xshift_i,
        'yshift': yshift_i,
    }


def extract_aperture_photometry(file_list, xcList, ycList, aperture_radius, use_parallel=True):
    """
    Extract photometry from all files for given aperture positions.
    
    Parameters
    ----------
    file_list : list
        List of FITS file paths
    xcList : array-like
        X coordinates of aperture centers
    ycList : array-like
        Y coordinates of aperture centers
    aperture_radius : float
        Radius of aperture in pixels
    use_parallel : bool, optional
        Whether to use parallel processing
        
    Returns
    -------
    dict
        Dictionary containing:
        - fluxList: Combined flux measurements
        - fluxErrList: Combined flux errors
        - timeList: Combined time measurements
        - xshiftList: Combined X shifts
        - yshiftList: Combined Y shifts
    """
    if use_parallel:
        # Process files in parallel
        results = Parallel(n_jobs=mp.cpu_count())(
            delayed(process_single_file)(fn, xcList, ycList, aperture_radius) for fn in file_list
        )
    else:
        # Process files sequentially
        results = [process_single_file(fn, xcList, ycList, aperture_radius) for fn in file_list]
    
    # Combine results
    fluxList = np.vstack([r['flux'] for r in results])
    fluxErrList = np.vstack([r['flux_err'] for r in results])
    timeList = np.concatenate([r['time'] for r in results])
    xshiftList = np.concatenate([r['xshift'] for r in results])
    yshiftList = np.concatenate([r['yshift'] for r in results])
    
    return {
        'fluxList': fluxList,
        'fluxErrList': fluxErrList,
        'timeList': timeList,
        'xshiftList': xshiftList,
        'yshiftList': yshiftList
    }


def calculate_planet_contribution(xcList, ycList, aperture_radius, psf_model_path):
    """
    Calculate planet contribution in each aperture using a PSF model.
    
    Parameters
    ----------
    xcList : array-like
        X coordinates of aperture centers
    ycList : array-like
        Y coordinates of aperture centers
    aperture_radius : float
        Radius of aperture in pixels
    psf_model_path : str
        Path to PSF model file
        
    Returns
    -------
    array-like
        Planet contribution factor for each aperture
    """
    # Load PSF model
    psf_model = fits.getdata(psf_model_path)
    
    # Calculate flux in each aperture
    planet_fm_flux_list = np.zeros_like(xcList, dtype=float)
    for i in range(len(xcList)):
        xc = xcList[i]
        yc = ycList[i]
        fi, _ = planet_aperture_photometry(
            psf_model, 
            psf_model*0.01,
            x_center=xc, 
            y_center=yc, 
            aperture_radius=aperture_radius
        )
        planet_fm_flux_list[i] = fi
    del psf_model
    gc.collect()
    
    return planet_fm_flux_list