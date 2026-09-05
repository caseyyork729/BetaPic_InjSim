#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation module for exoplanet light curve analysis.

This module validates the systematic error correction by applying the same correction
process to validation apertures that should contain only star light and systematic noise.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d as gf
import h5py

from BetaPic.MULT_DirectPlanetLightCurveAnalysis.utils import calculate_planet_pixels, save_to_hdf5
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.aperture_photometry import extract_aperture_photometry, planet_aperture_photometry
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.visualization import plot_aperture_positions, plot_raw_lightcurves, visualize_all_apertures
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.modeling import lc_model, fit_lightcurve, visualize_results


def calculate_validation_positions(roll1, roll2, pixel_scale, validation_pa_list, 
                                  validation_separation_list, star_x_center, star_y_center):
    """
    Calculate positions for validation apertures.
    
    Parameters
    ----------
    roll1, roll2 : float
        Roll angles for the two telescope orientations
    pixel_scale : float
        Pixel scale in arcseconds/pixel
    validation_pa_list : list
        List of position angles for validation apertures
    validation_separation_list : list
        List of separations for validation apertures in milliarcseconds
    star_x_center, star_y_center : float
        Star center coordinates in pixels
        
    Returns
    -------
    tuple
        ((xc_list1, yc_list1), (xc_list2, yc_list2)) - Coordinates for both rolls
    """
    xc_val_list1 = np.zeros(len(validation_pa_list))
    yc_val_list1 = np.zeros(len(validation_pa_list))
    xc_val_list2 = np.zeros(len(validation_pa_list))
    yc_val_list2 = np.zeros(len(validation_pa_list))
    
    # Calculate positions for both rolls
    for i, pa in enumerate(validation_pa_list):
        # Calculate the positions in pixels
        val_pos1 = calculate_planet_pixels(validation_separation_list[i], pa, roll1, pixel_scale,
                                         star_x_center=star_x_center, star_y_center=star_y_center)
        val_pos2 = calculate_planet_pixels(validation_separation_list[i], pa, roll2, pixel_scale,
                                         star_x_center=star_x_center, star_y_center=star_y_center)
        
        xc_val_list1[i] = val_pos1[0]
        yc_val_list1[i] = val_pos1[1]
        xc_val_list2[i] = val_pos2[0]
        yc_val_list2[i] = val_pos2[1]
    
    return (xc_val_list1, yc_val_list1), (xc_val_list2, yc_val_list2)



def process_validation_apertures(database, key, roll_idx, validation_pos, 
                               planet_pos, comp_pos, aperture_radius, star_x_center, star_y_center,
                               plot_dir, use_parallel=True):
    """
    Process validation apertures for a specific telescope roll.
    
    Parameters
    ----------
    database : object
        Database object
    key : str
        Database key
    roll_idx : int
        Roll index (0 or 1)
    validation_pos : tuple
        Validation positions for both rolls
    planet_pos : tuple
        Planet positions for both rolls
    comp_pos : tuple
        Comparison positions for both rolls
    aperture_radius : float
        Aperture radius in pixels
    plot_dir : str
        Directory for saving plots
    use_parallel : bool, optional
        Whether to use parallel processing
        
    Returns
    -------
    dict
        Extracted data for validation apertures
    """
    print(f"\nProcessing Validation Apertures for Roll {roll_idx}...")
    
    # Select appropriate file list and positions
    if roll_idx == 0:
        file_list = database.obs[key]['FITSFILE'].value[0:10]
        xc_planet, yc_planet = planet_pos[0]
        xc_comp_list, yc_comp_list = comp_pos[0]
        xc_val_list, yc_val_list = validation_pos[0]
    else:
        file_list = database.obs[key]['FITSFILE'].value[10:20]
        xc_planet, yc_planet = planet_pos[1]
        xc_comp_list, yc_comp_list = comp_pos[1]
        xc_val_list, yc_val_list = validation_pos[1]
    
    # Load first image for visualization
    im = np.mean(fits.getdata(file_list[0], ext=1), axis=0)
    
    # Visualize all apertures
    comp_positions = list(zip(xc_comp_list, yc_comp_list))
    val_positions = list(zip(xc_val_list, yc_val_list))
    
    visualize_all_apertures(
        im, (xc_planet, yc_planet), comp_positions, val_positions,
        aperture_radius, star_x_center, star_y_center,
        title=f"All Aperture Positions - Roll {roll_idx}",
        output_path=os.path.join(plot_dir, 'apertures', f'all_apertures_roll{roll_idx}.png')
    )
    
    # Combine all coordinates for photometry
    xcList = np.concatenate((np.array([xc_planet]), xc_comp_list, xc_val_list))
    ycList = np.concatenate((np.array([yc_planet]), yc_comp_list, yc_val_list))
    
    # Extract photometry for all apertures
    data = extract_aperture_photometry(
        file_list, xcList, ycList, aperture_radius, 
        use_parallel=use_parallel
    )
    
    # Number of apertures of each type
    n_planet = 1
    n_comp = len(xc_comp_list)
    n_val = len(xc_val_list)
    
    # Create a data dictionary with aperture types
    result_data = {
        'fluxList': data['fluxList'],
        'fluxErrList': data['fluxErrList'],
        'timeList': data['timeList'],
        'xshiftList': data['xshiftList'],
        'yshiftList': data['yshiftList'],
        'xc': xcList,
        'yc': ycList,
        'aperture_radius': aperture_radius,
        'n_planet': n_planet,
        'n_comp': n_comp,
        'n_val': n_val
    }
    
    return result_data


def validate_systematic_correction(validation_data, pca_components, stellar_model, 
                                  roll_idx, plot_dir, filter_name, config, save_samples=10):
    """
    Validate systematic correction by applying it to validation apertures.
    
    Parameters
    ----------
    validation_data : dict
        Data extracted from validation apertures
    pca_components : ndarray
        PCA components of systematic variations
    stellar_model : array-like
        Stellar light curve model
    roll_idx : int
        Roll index
    plot_dir : str
        Directory for saving plots
    filter_name : str
        Filter name
        
    Returns
    -------
    dict
        Validation results
    """
    print(f"\nValidating Systematic Correction for Roll {roll_idx}...")
    
    # Get data
    fluxList = validation_data['fluxList']
    fluxErrList = validation_data['fluxErrList']
    timeList = validation_data['timeList']
    n_planet = validation_data['n_planet']
    n_comp = validation_data['n_comp']
    nValidation = fluxList.shape[1] - n_planet - n_comp
    lcList = np.zeros((fluxList.shape[0], nValidation), dtype=float)
    lcErrList = np.zeros((fluxList.shape[0], nValidation), dtype=float)
    corrected_lcList = np.zeros((fluxList.shape[0], nValidation), dtype=float)
    corrected_lcErrList = np.zeros((fluxList.shape[0], nValidation), dtype=float)    
    corrected_lcSampleList = np.zeros((fluxList.shape[0], save_samples, nValidation), dtype=float)
    
    # Validation apertures start after planet and comparison apertures
    validation_start_idx = n_planet + n_comp
    
    # Convert time to hours for model
    t0 = timeList[0]
    t_hours = (timeList - t0) * 24
    
    # Select principal components
    PC1 = pca_components[:, 0]
    PC2 = pca_components[:, 1]
    PC3 = pca_components[:, 2]
    
    # Create arrays to store validation results
    n_validation = fluxList.shape[1] - validation_start_idx
    best_params_list = []
    validation_qualities = []
    
    # Output directory
    output_dir = os.path.join(plot_dir, 'validation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each validation aperture
    for i in range(n_validation):
        val_idx = validation_start_idx + i
        
        # Normalize flux
        flux_i = fluxList[:, val_idx]
        flux_err_i = fluxErrList[:, val_idx]
        norm = np.median(flux_i)
        f_norm = flux_i / norm
        ferr_norm = flux_err_i / norm
        
        # For validation apertures, we assume there's no planet contribution
        # (these are chosen to be away from the planet)
        planet_factor = 0.0
        
        # Apply the same fitting procedure as for the planet
        try:
            # Try with default MCMC parameters
            samples, best_params = fit_lightcurve(
                t_hours, f_norm, stellar_model, PC1, PC2, PC3, 
                yerr=ferr_norm, priors=config.LC_PRIORS,
                nwalkers=config.MCMC_WALKERS // 2, nsteps=config.MCMC_STEPS // 2, discard=config.MCMC_BURNIN,
                fit_planet_factor=config.FIT_PLANET_FACTOR_VALIDATION,
            )
            
            # Visualize the fit
            visualize_results(
                t_hours, f_norm, ferr_norm, best_params, stellar_model, PC1, PC2, PC3, 
                samples=samples,
                output_path=os.path.join(output_dir, f'{filter_name}_roll{roll_idx}_validation{i+1}_fit.png'),
                fit_planet_factor=config.FIT_PLANET_FACTOR_VALIDATION
            )
            
            # Calculate correction quality (improvement in RMS)
            # Calculate model
            model = lc_model(t_hours, best_params, stellar_model, PC1, PC2, PC3, fit_planet_factor=config.FIT_PLANET_FACTOR_VALIDATION)
            
            # Calculate residuals before and after correction
            residuals_before = f_norm - stellar_model
            residuals_after = f_norm - model
            
            rms_before = np.std(residuals_before)
            rms_after = np.std(residuals_after)
            improvement = rms_before / rms_after
            
            # Store results
            best_params_list.append(best_params)
            validation_qualities.append(improvement)
            lcList[:, i] = f_norm
            lcErrList[:, i] = ferr_norm
            systematic_comp_model = best_params[3] + best_params[4] * PC1 + best_params[5] * PC2 + best_params[6] * PC3
            # take out the stellar variability part, assuming all flux are stellar for the comparison apertures
            
            planet_factor_bestfit = best_params[-1] if config.FIT_PLANET_FACTOR_VALIDATION else planet_factor
            stellar_lc_factor = (1 - planet_factor_bestfit) * stellar_model
            planet_lc_corrected = (f_norm / systematic_comp_model - stellar_lc_factor) / planet_factor_bestfit
            
            corrected_lcList[:, i] = planet_lc_corrected
            corrected_lcErrList[:, i] = ferr_norm / np.abs(systematic_comp_model) / planet_factor_bestfit

            # Save samples for further analysis
            if save_samples > 0:
                sample_index = np.random.randint(0, len(samples), size=save_samples)
                for j, samp_j in enumerate(sample_index):
                    sample_params = samples[samp_j]
                    systematic_comp_model_sample = sample_params[3] + \
                        sample_params[4] * PC1 + sample_params[5] * PC2 + sample_params[6] * PC3
                    stellar_lc_factor_sample = (1 - sample_params[-1]) * stellar_model
                    planet_lc_corrected_sample = (f_norm / systematic_comp_model_sample - stellar_lc_factor_sample) / sample_params[-1]
                    corrected_lcSampleList[:, j, i] = planet_lc_corrected_sample
                    
            # Print results
            print(f"Validation Aperture {i+1}:")
            print(f"  Before correction: RMS = {rms_before:.5f}")
            print(f"  After correction: RMS = {rms_after:.5f}")
            print(f"  Improvement: {improvement:.2f}x")
            print("  Best-fit parameters:")
            param_names = ['A', 'B', 'P', 'w0', 'w1', 'w2', 'w3']
            for name, val in zip(param_names, best_params):
                print(f"    {name}: {val:.6f}")
            
        except Exception as e:
            print(f"Error fitting validation aperture {i+1}: {e}")
            best_params_list.append(None)
            validation_qualities.append(0.0)

    # Save results to HDF5
    output_file = os.path.join(output_dir, f'Validation_Light_Curve_roll{roll_idx}.h5')
    with h5py.File(output_file, 'w') as f:    
        f.create_dataset('timeList', data=timeList)
        f.create_dataset('lightcurveList', data=lcList)
        f.create_dataset('errorList', data=lcErrList)
        f.create_dataset('corrected_lightcurveList', data=corrected_lcList)
        f.create_dataset('corrected_errorList', data=corrected_lcErrList)        
        f.create_dataset('corrected_lightcurveSampleList', data=corrected_lcSampleList)
    
    return {
        'best_params': best_params_list,
        'correction_quality': np.array(validation_qualities),
        'timeList': timeList
    }