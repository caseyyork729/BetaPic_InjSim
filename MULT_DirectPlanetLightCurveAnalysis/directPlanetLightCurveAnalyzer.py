#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for analyzing time-series images of exoplanets.

This script extracts light curves, removes systematic noise, and identifies signals
from time-series images of an exoplanet.
"""

FilterName = 'F210M'  # Set to None to disable filter selection and load all filters

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
from scipy.ndimage import gaussian_filter1d as gf
import spaceKLIP

# Import local modules
if FilterName == 'F410M':
    import BetaPic.MULT_DirectPlanetLightCurveAnalysis.config_F410M as config
elif FilterName == 'F210M':
    import BetaPic.MULT_DirectPlanetLightCurveAnalysis.config_F210M as config
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.utils import (create_directories, calculate_planet_pixels, 
                  pca_timeseries, save_to_hdf5, load_from_hdf5)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.aperture_photometry import (extract_aperture_photometry, calculate_planet_contribution)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.modeling import (fit_lightcurve, visualize_results, fit_fourier, visualize_fourier_fit)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.visualization import (plot_aperture_positions, visualize_all_apertures, plot_raw_lightcurves, 
                         plot_systematic_models, plot_pca_components, plot_stellar_model)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.validation import (calculate_validation_positions, process_validation_apertures,
                      validate_systematic_correction, visualize_all_apertures)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.BetaPicStellarLCModel import BetaPicStellarLC


def setup_directories():
    """Create necessary directories for output files."""
    dirs = [
        config.LIGHTCURVE_DIR,
        config.PLOT_DIR,
        os.path.join(config.PLOT_DIR, 'apertures'),
        os.path.join(config.PLOT_DIR, 'lightcurves'),
        os.path.join(config.PLOT_DIR, 'pca'),
        os.path.join(config.PLOT_DIR, 'modeling'),
        os.path.join(config.PLOT_DIR, 'validation'),
    ]
    create_directories(dirs)


def initialize_database():
    """Initialize the database and return key information."""
    # Create database
    database = spaceKLIP.database.create_database(
        input_dir=os.path.join(config.DATA_ROOT, 'aligned'),
        file_type='calints.fits',
        output_dir=config.DATA_ROOT,                                    
        pid=config.PID,
        verbose=False
    )
    
    # Get key for FILTER
    key = config.ObservationKey
    
    # Print database info
    print("Database Summary:")
    print('Available keys:', database.obs[key].keys())
    
    # Get example file info
    f0 = database.obs[key]['FITSFILE'].value[0]
    with fits.open(f0) as hdul:
        print(hdul.info())
    
    return database, key


def calculate_planet_positions(database, key):
    """Calculate planet positions for both telescope rolls."""
    pixel_scale = database.obs[key]['PIXSCALE'].value[0]
    
    # Get filter configuration
    filter_config = config.FILTER_CONFIGS[config.FILTER]
    
    # Calculate separation and PA values
    sep1_value = filter_config['separation']
    sep1_err = filter_config['separation_err']
    sep2_value = filter_config['separation']
    sep2_err = filter_config['separation_err']
    
    pa1_value = filter_config['pa']
    pa1_err = filter_config['pa_err']
    pa2_value = filter_config['pa']
    pa2_err = filter_config['pa_err']
    
    print(f"Separation1: {sep1_value:.2f} ± {sep1_err:.2f} mas")
    print(f"Separation2: {sep2_value:.2f} ± {sep2_err:.2f} mas")
    print(f"PA1: {pa1_value:.2f} ± {pa1_err:.2f} deg")
    print(f"PA2: {pa2_value:.2f} ± {pa2_err:.2f} deg")
    
    # Get roll angles
    roll1List = database.obs[key]['ROLL_REF'].value[0:10]
    roll2List = database.obs[key]['ROLL_REF'].value[10:20]
    
    roll1 = np.mean(roll1List)
    roll1_err = np.std(roll1List)
    roll2 = np.mean(roll2List)
    roll2_err = np.std(roll2List)
    
    print(f"Roll1: {roll1:.5f} ± {roll1_err:.5f}")
    print(f"Roll2: {roll2:.5f} ± {roll2_err:.5f}")
    
    # Calculate planet positions with uncertainties
    planet1_pos = calculate_planet_pixels(sep1_value, pa1_value, roll1, pixel_scale, 
                                        sep_err=sep1_err, pa_err=pa1_err, roll_err=roll1_err,
                                        star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER)
    
    planet2_pos = calculate_planet_pixels(sep2_value, pa2_value, roll2, pixel_scale, 
                                        sep_err=sep2_err, pa_err=pa2_err, roll_err=roll2_err,
                                        star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER)
    
    # Extract positions and errors
    xc1, yc1 = planet1_pos['x'], planet1_pos['y']
    xc1_err, yc1_err = planet1_pos['x_err'], planet1_pos['y_err']
    xc2, yc2 = planet2_pos['x'], planet2_pos['y']
    xc2_err, yc2_err = planet2_pos['x_err'], planet2_pos['y_err']
    
    # Print the results
    print("\nPlanet Pixel Coordinates:")
    print(f"Roll 1: ({xc1:.3f} ± {xc1_err:.3f}, {yc1:.3f} ± {yc1_err:.3f})")
    print(f"Roll 2: ({xc2:.3f} ± {xc2_err:.3f}, {yc2:.3f} ± {yc2_err:.3f})")
    
    return (xc1, yc1), (xc2, yc2), roll1, roll2


def calculate_comparison_positions(roll1, roll2, pixel_scale):
    """Calculate positions for comparison apertures."""
    xc_comp_list1 = np.zeros(len(config.COMPARISON_PA_LIST))
    yc_comp_list1 = np.zeros(len(config.COMPARISON_PA_LIST))
    xc_comp_list2 = np.zeros(len(config.COMPARISON_PA_LIST))
    yc_comp_list2 = np.zeros(len(config.COMPARISON_PA_LIST))
    
    # Calculate positions for both rolls
    for i, pa in enumerate(config.COMPARISON_PA_LIST):
        # Calculate positions in pixels
        planet_pos1 = calculate_planet_pixels(config.COMPARISON_SEPARATION_LIST[i], pa, roll1, pixel_scale,
                                           star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER)
        planet_pos2 = calculate_planet_pixels(config.COMPARISON_SEPARATION_LIST[i], pa, roll2, pixel_scale,
                                           star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER)
        
        xc_comp_list1[i] = planet_pos1[0]
        yc_comp_list1[i] = planet_pos1[1]
        xc_comp_list2[i] = planet_pos2[0]
        yc_comp_list2[i] = planet_pos2[1]
    
    return (xc_comp_list1, yc_comp_list1), (xc_comp_list2, yc_comp_list2)


def visualize_aperture_positions(database, key, planet_pos, comp_pos, validation_pos=None):
    """Visualize aperture positions on the images."""
    # Load images
    fn1 = database.obs[key]['FITSFILE'].value[0]
    fn2 = database.obs[key]['FITSFILE'].value[10]
    
    im1 = np.mean(fits.getdata(fn1, ext=1), axis=0)
    im2 = np.mean(fits.getdata(fn2, ext=1), axis=0)
    
    # Create plots
    (xc1, yc1), (xc2, yc2) = planet_pos
    (xc_comp_list1, yc_comp_list1), (xc_comp_list2, yc_comp_list2) = comp_pos
    
    # If validation positions are provided
    if validation_pos and config.ENABLE_VALIDATION:
        (xc_val_list1, yc_val_list1), (xc_val_list2, yc_val_list2) = validation_pos
        
        # Create combined visualizations with all aperture types
        comp_pos1 = list(zip(xc_comp_list1, yc_comp_list1))
        val_pos1 = list(zip(xc_val_list1, yc_val_list1))
        visualize_all_apertures(
            im1, (xc1, yc1), comp_pos1, val_pos1, config.APERTURE_RADIUS,
            star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER,
            title=f"All Aperture Positions - Roll 1 ({config.FILTER})",
            output_path=os.path.join(config.PLOT_DIR, 'apertures', f'{config.FILTER}_all_apertures_roll1.png')
        )
        
        comp_pos2 = list(zip(xc_comp_list2, yc_comp_list2))
        val_pos2 = list(zip(xc_val_list2, yc_val_list2))
        visualize_all_apertures(
            im2, (xc2, yc2), comp_pos2, val_pos2, config.APERTURE_RADIUS,
            star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER,
            title=f"All Aperture Positions - Roll 2 ({config.FILTER})",
            output_path=os.path.join(config.PLOT_DIR, 'apertures', f'{config.FILTER}_all_apertures_roll2.png')
        )
    else:
        # Plot for roll 1
        comp_pos1 = list(zip(xc_comp_list1, yc_comp_list1))
        plot_aperture_positions(
            im1, (xc1, yc1), comp_pos1, config.APERTURE_RADIUS,
            star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER,
            title=f"Aperture Positions - Roll 1 ({config.FILTER})",
            output_path=os.path.join(config.PLOT_DIR, 'apertures', f'{config.FILTER}_roll1_apertures.png')
        )
        
        # Plot for roll 2
        comp_pos2 = list(zip(xc_comp_list2, yc_comp_list2))
        plot_aperture_positions(
            im2, (xc2, yc2), comp_pos2, config.APERTURE_RADIUS,
            star_x_center=config.STAR_X_CENTER, star_y_center=config.STAR_Y_CENTER,
            title=f"Aperture Positions - Roll 2 ({config.FILTER})",
            output_path=os.path.join(config.PLOT_DIR, 'apertures', f'{config.FILTER}_roll2_apertures.png')
        )


def process_roll(database, key, roll_idx, planet_pos, comp_pos, aperture_radius):
    """
    Process data for a specific telescope roll.
    
    Parameters
    ----------
    database : object
        Database object
    key : str
        Database key
    roll_idx : int
        Roll index (0 or 1)
    planet_pos : tuple
        Planet positions for both rolls
    comp_pos : tuple
        Comparison positions for both rolls
    aperture_radius : float
        Aperture radius in pixels
        
    Returns
    -------
    dict
        Extracted data
    """
    print(f"\nProcessing Roll {roll_idx}...")
    
    # Select appropriate file list and positions
    if roll_idx == 0:
        file_list = database.obs[key]['FITSFILE'].value[0:10]
        xc, yc = planet_pos[0]
        xc_comp_list, yc_comp_list = comp_pos[0]
    else:
        file_list = database.obs[key]['FITSFILE'].value[10:20]
        xc, yc = planet_pos[1]
        xc_comp_list, yc_comp_list = comp_pos[1]
    
    # Combine planet and comparison coordinates
    xcList = np.concatenate((np.array([xc]), xc_comp_list))
    ycList = np.concatenate((np.array([yc]), yc_comp_list))
    
    # Extract photometry
    data = extract_aperture_photometry(
        file_list, xcList, ycList, aperture_radius, 
        use_parallel=config.USE_PARALLEL
    )
    
    # Calculate planet contribution
    # Get PSF model path from the first file
    psf_model_path = os.path.join(
        config.PLANET_CONTRIBUTION_DIR, 
        os.path.basename(file_list[0]).replace('.fits', '_psf_model.fits')
    )
    
    if os.path.exists(psf_model_path):
        planet_fm_flux_list = calculate_planet_contribution(
            xcList, ycList, aperture_radius, psf_model_path
        )
        
        # Calculate planet contribution factors
        planet_contribution = np.zeros_like(planet_fm_flux_list, dtype=float)
        for j in range(len(xcList)):
            normalization = np.median(data['fluxList'][:, j])
            planet_contribution[j] = planet_fm_flux_list[j] / normalization
            print(f'In aperture {j}, the planet contribution is {planet_contribution[j]:.4f}')
    else:
        print(f"Warning: PSF model file not found: {psf_model_path}")
        planet_fm_flux_list = np.zeros_like(xcList, dtype=float)
        planet_contribution = np.zeros_like(xcList, dtype=float)
    
    # Create a complete data dictionary
    result_data = {
        'fluxList': data['fluxList'],
        'fluxErrList': data['fluxErrList'],
        'timeList': data['timeList'],
        'xshiftList': data['xshiftList'],
        'yshiftList': data['yshiftList'],
        'planet_fm_flux': planet_fm_flux_list,
        'planet_contribution': planet_contribution,
        'xc': xcList,
        'yc': ycList,
        'aperture_radius': aperture_radius
    }
    
    # Save data to HDF5
    save_filename = os.path.join(
        config.LIGHTCURVE_DIR, 
        f'Planet_LightCurves_roll_{roll_idx}_aper_{aperture_radius:.1f}.hdf5'
    )
    save_to_hdf5(save_filename, result_data)
    
    # Plot raw light curves
    plot_raw_lightcurves(
        data['timeList'], data['fluxList'], data['fluxErrList'], 
        planet_contribution=planet_contribution,
        output_path=os.path.join(
            config.PLOT_DIR, 'lightcurves', 
            f'{config.FILTER}_roll{roll_idx}_raw_lightcurves.png'
        )
    )
    
    return result_data


def remove_systematics(data, roll_idx, stellar_model=None):
    """
    Remove systematic noise from the light curve.
    
    Parameters
    ----------
    data : dict
        Extracted photometry data
    roll_idx : int
        Roll index
    stellar_model : array-like, optional
        Stellar light curve model
        
    Returns
    -------
    tuple
        (corrected_lc, systematic_model, astrophysical_model)
    """
    print(f"\nRemoving systematics for Roll {roll_idx}...")
    
    # Get data
    timeList = data['timeList']
    fluxList = data['fluxList']
    fluxErrList = data['fluxErrList']
    # Create empty array to store corrected light curves
    lcList = np.zeros_like(fluxList, dtype=float)
    lcErrList = np.zeros_like(fluxList, dtype=float)
    corrected_lcList = np.zeros_like(fluxList, dtype=float)
    corrected_lcErrList = np.zeros_like(fluxList, dtype=float)    
    planet_contribution = data['planet_contribution']
    
    # Get or calculate stellar model
    if stellar_model is None:
        stellar_lc = BetaPicStellarLC(timeList)
        if FilterName == 'F410M':
            stellar_model = stellar_lc.F410M_LC
        elif FilterName == 'F210M':
            stellar_model = stellar_lc.F210M_LC
    
    # Create array of systematic models from comparison apertures
    systematic_model_array = np.zeros((fluxList.shape[0], fluxList.shape[1]-1), dtype=float)
    
    # Plot stellar model comparison with a reference aperture
    for aperture_idx in range(1, fluxList.shape[1]):        
        plot_stellar_model(
            timeList, stellar_model, 
            comparison_flux=fluxList[:, aperture_idx], aperture_idx=aperture_idx,
            filter_sigma=config.FILTER_SIGMA,
            output_path=os.path.join(
                config.PLOT_DIR, 'modeling', 
                f'{config.FILTER}_roll{roll_idx}_aperture_{aperture_idx}_model.png'
            )
        )
    
    # Extract systematic model from each comparison aperture
    for aperture_i in range(1, fluxList.shape[1]):
        flux_i = fluxList[:, aperture_i]
        norm = np.median(flux_i)
        f_norm = flux_i / norm
        systematic_model_array[:, aperture_i-1] = gf(f_norm / stellar_model, config.FILTER_SIGMA)
    
    # Plot systematic models
    plot_systematic_models(
        timeList, systematic_model_array, 
        planet_flux=fluxList[:, 0],
        sigma=config.FILTER_SIGMA,
        output_path=os.path.join(
            config.PLOT_DIR, 'modeling', 
            f'{config.FILTER}_roll{roll_idx}_planet_systematic_models.png'
        )
    )

    
    # Perform PCA on systematic models
    systematic_PCs = pca_timeseries(systematic_model_array, n_components=config.N_COMPONENTS)
    
    # Plot PCA components
    plot_pca_components(
        timeList, systematic_PCs,
        n_components=min(3, config.N_COMPONENTS),
        output_path=os.path.join(
            config.PLOT_DIR, 'pca', 
            f'{config.FILTER}_roll{roll_idx}_pca_components.png'
        )
    )
    
    # Fit the planet light curve with systematic model
    t0 = timeList[0]
    t_hours = (timeList - t0) * 24  # convert to hours
    aperture_i = 0
    planet_factor = planet_contribution[aperture_i]
    print(f'Planet contribution in aperture {aperture_i}: {planet_factor:.4f}')
    
    # Prepare data for fitting
    lc = fluxList[:, aperture_i] / np.median(fluxList[:, aperture_i])
    lc_err = fluxErrList[:, aperture_i] / np.median(fluxList[:, aperture_i])
    
    # Select principal components
    PC1 = systematic_PCs[:, 0]
    PC2 = systematic_PCs[:, 1]
    PC3 = systematic_PCs[:, 2]
    
    # Fit the light curve
    samples, best_params = fit_lightcurve(
        t_hours, lc, stellar_model, PC1, PC2, PC3, 
        yerr=lc_err, priors=config.LC_PRIORS,
        nwalkers=config.MCMC_WALKERS, nsteps=config.MCMC_STEPS, discard=config.MCMC_BURNIN,
        planet_factor_fixed=planet_factor,
        fit_planet_factor=config.FIT_PLANET_FACTOR
    )
    
    # Visualize the fit
    visualize_results(
        t_hours, lc, lc_err, best_params, stellar_model, PC1, PC2, PC3, 
        samples=samples,
        fit_planet_factor=config.FIT_PLANET_FACTOR,
        output_path=os.path.join(
            config.PLOT_DIR, 'modeling', 
            f'{config.FILTER}_roll{roll_idx}_lightcurve_fit_planet.png'
        )
    )

    # Print best-fit parameters
    print("\nBest-fit parameters:")
    param_names = ['A', 'B', 'P', 'w0', 'w1', 'w2', 'w3', 'planet_factor']
    for name, val in zip(param_names, best_params):
        print(f"{name}: {val:.6f}")
    
    # Extract models
    planet_factor_bestfit = best_params[7]
    systematic_model = best_params[3] + best_params[4] * PC1 + best_params[5] * PC2 + best_params[6] * PC3
    stellar_lc_factor = (1 - planet_factor_bestfit) * stellar_model
    astrophysical_model = stellar_lc_factor + planet_factor_bestfit * (
        1 + best_params[0] * np.sin(2*np.pi*t_hours/best_params[2]) + 
        best_params[1] * np.cos(2*np.pi*t_hours/best_params[2])
    )
    
    # Calculate planet light curve corrected for systematics    
    planet_lc_corrected = (lc / systematic_model - stellar_lc_factor) / planet_factor_bestfit
    lcList[:, aperture_i] = lc
    lcErrList[:, aperture_i] = lc_err
    corrected_lcList[:, aperture_i] = planet_lc_corrected
    corrected_lcErrList[:, aperture_i] = lc_err / np.abs(systematic_model) / planet_factor_bestfit
    
    # Save corrected light curve
    save_fn = os.path.join(
        config.LIGHTCURVE_DIR, 
        f'Planet_LightCurve_Corrected_Roll_{roll_idx}.txt'
    )
    np.savetxt(save_fn, np.array([timeList, planet_lc_corrected]).T)

    # perform the same fit to the comparison apertures
    for aperture_i in range(1, fluxList.shape[1]):
        planet_factor_comp = planet_contribution[aperture_i]        
        print(f'Planet contribution in aperture {aperture_i}: {planet_factor_comp:.4f}')
        
        # Prepare data for fitting
        lc_comp = fluxList[:, aperture_i] / np.median(fluxList[:, aperture_i])
        lc_err_comp = fluxErrList[:, aperture_i] / np.median(fluxList[:, aperture_i])
        
        # Fit the light curve
        samples_comp, best_params_comp = fit_lightcurve(
            t_hours, lc_comp, stellar_model, PC1, PC2, PC3, 
            yerr=lc_err_comp, priors=config.LC_PRIORS,
            nwalkers=config.MCMC_WALKERS, nsteps=config.MCMC_STEPS, discard=config.MCMC_BURNIN,
            planet_factor_fixed=planet_factor_comp,
            fit_planet_factor=config.FIT_PLANET_FACTOR
        )
        
        # Visualize the fit
        visualize_results(
            t_hours, lc_comp, lc_err_comp, best_params_comp, stellar_model, PC1, PC2, PC3, 
            samples=samples_comp,
            fit_planet_factor=config.FIT_PLANET_FACTOR,
            output_path=os.path.join(
                config.PLOT_DIR, 'modeling', 
                f'{config.FILTER}_roll{roll_idx}_lightcurve_fit_aperture_{aperture_i}.png'
            )
        )
        lcList[:, aperture_i] = lc_comp
        lcErrList[:, aperture_i] = lc_err_comp
        # planet_factor_bestfit_comp = best_params_comp[7]
        systematic_comp_model = best_params_comp[3] + best_params_comp[4] * PC1 + best_params_comp[5] * PC2 + best_params_comp[6] * PC3
        # take out the stellar variability part, assuming all flux are stellar for the comparison apertures
        lc_comp_corrected = lc_comp / systematic_comp_model / stellar_model
        lc_comp_err_corrected = lc_err_comp / np.abs(systematic_comp_model) / stellar_model
        corrected_lcList[:, aperture_i] = lc_comp_corrected
        corrected_lcErrList[:, aperture_i] = lc_comp_err_corrected

    # save the light curves in to a hdf5 file at the config.PLOT_DIR/lightcurves
    save_fn = os.path.join(
        config.PLOT_DIR, 'lightcurves',
        f'Light_Curves_Roll_{roll_idx}.hdf5')
    # create the hdf5 file
    with h5py.File(save_fn, 'w') as f:
        # create the datasets
        f.create_dataset('timeList', data=timeList)
        f.create_dataset('lightcurveList', data=lcList)
        f.create_dataset('errorList', data=lcErrList)
        f.create_dataset('corrected_lightcurveList', data=corrected_lcList)
        f.create_dataset('corrected_errorList', data=corrected_lcErrList)
        f.create_dataset('systematicModel', data=systematic_model_array)        


    
    return planet_lc_corrected, systematic_model, astrophysical_model, systematic_PCs


def combine_light_curves(data_roll0, data_roll1):
    """
    Combine light curves from both rolls and fit a Fourier model.
    
    Parameters
    ----------
    data_roll0, data_roll1 : dict
        Data from both rolls
        
    Returns
    -------
    tuple
        (samples, best_params, ls_params)
    """
    print("\nCombining light curves from both rolls...")
    
    # Load corrected light curves
    fn_roll0 = os.path.join(config.LIGHTCURVE_DIR, f'Planet_LightCurve_Corrected_Roll_0.txt')
    fn_roll1 = os.path.join(config.LIGHTCURVE_DIR, f'Planet_LightCurve_Corrected_Roll_1.txt')
    
    t1, lc1 = np.loadtxt(fn_roll0, unpack=True)
    t2, lc2 = np.loadtxt(fn_roll1, unpack=True)
    
    # Combine data
    t = np.concatenate([t1, t2])
    t0 = t1[0]  # Reference time
    t_hours = (t - t0) * 24  # Convert to hours
    lc = np.concatenate([lc1, lc2])
    
    # Fit Fourier model
    samples, best_params, ls_params = fit_fourier(
        t_hours, lc, priors=config.FOURIER_PRIORS,
        nwalkers=config.MCMC_WALKERS, nsteps=config.MCMC_STEPS, discard=config.MCMC_BURNIN
    )
    
    # Visualize fit
    visualize_fourier_fit(
        t_hours, lc, samples, best_params, ls_params,
        output_path=os.path.join(
            config.PLOT_DIR, 'modeling', 
            f'{config.FILTER}_combined_fourier_fit.png'
        )
    )
    
    return samples, best_params, ls_params


def main():
    """Main execution function."""
    print(f"Starting exoplanet analysis for filter: {config.FILTER}")
    
    # Setup directories
    setup_directories()
    
    # Initialize database
    database, key = initialize_database()
    
    # Get pixel scale
    pixel_scale = database.obs[key]['PIXSCALE'].value[0]
    
    # Calculate planet positions
    planet_pos, planet_pos2, roll1, roll2 = calculate_planet_positions(database, key)
    
    # Calculate comparison positions
    comp_pos = calculate_comparison_positions(roll1, roll2, pixel_scale)
       # Calculate validation positions if enabled
    if config.ENABLE_VALIDATION:
        validation_pos = calculate_validation_positions(
            roll1, roll2, pixel_scale, 
            config.VALIDATION_PA_LIST, config.VALIDATION_SEPARATION_LIST,
            config.STAR_X_CENTER, config.STAR_Y_CENTER
        )
    else:
        validation_pos = None
    
    # Visualize aperture positions
    visualize_aperture_positions(database, key, (planet_pos, planet_pos2), comp_pos, validation_pos)
    
    # Process each roll
    data_roll0 = process_roll(
        database, key, 0, (planet_pos, planet_pos2), comp_pos, 
        config.APERTURE_RADIUS
    )
    
    data_roll1 = process_roll(
        database, key, 1, (planet_pos, planet_pos2), comp_pos, 
        config.APERTURE_RADIUS
    )
    
    # Get stellar model
    stellar_lc = BetaPicStellarLC(data_roll0['timeList'])
    if FilterName == 'F410M':
        stellar_model0 = stellar_lc.F410M_LC
    elif FilterName == 'F210M':
        stellar_model0 = stellar_lc.F210M_LC
    
    stellar_lc = BetaPicStellarLC(data_roll1['timeList'])
    if FilterName == 'F410M':
        stellar_model1 = stellar_lc.F410M_LC
    elif FilterName == 'F210M':
        stellar_model1 = stellar_lc.F210M_LC
    
    # Remove systematics
    planet_lc0, systematic_model0, astrophysical_model0, systematic_PCs0 = remove_systematics(
        data_roll0, 0, stellar_model=stellar_model0
    )
    
    planet_lc1, systematic_model1, astrophysical_model1, systematic_PCs1 = remove_systematics(
        data_roll1, 1, stellar_model=stellar_model1
    )

    if config.ENABLE_VALIDATION:
        # Process validation apertures
        validation_data0 = process_validation_apertures(
            database, key, 0, validation_pos, (planet_pos, planet_pos2), comp_pos,
            config.APERTURE_RADIUS, config.STAR_X_CENTER, config.STAR_Y_CENTER,
            config.PLOT_DIR, use_parallel=config.USE_PARALLEL
        )
        
        validation_data1 = process_validation_apertures(
            database, key, 1, validation_pos, (planet_pos, planet_pos2), comp_pos,
            config.APERTURE_RADIUS, config.STAR_X_CENTER, config.STAR_Y_CENTER,
            config.PLOT_DIR, use_parallel=config.USE_PARALLEL
        )
        
        # Validate systematic correction
        validation_results0 = validate_systematic_correction(
            validation_data0, systematic_PCs0, stellar_model0, 0, config.PLOT_DIR, config.FILTER, config, 
            save_samples=config.SAVE_SAMPLES
        )
        
        validation_results1 = validate_systematic_correction(
            validation_data1, systematic_PCs1, stellar_model1, 1, config.PLOT_DIR, config.FILTER, config,
            save_samples=config.SAVE_SAMPLES
        )
        
        # Save validation results
        save_fn0 = os.path.join(config.LIGHTCURVE_DIR, f'Validation_Results_Roll_0.hdf5')
        save_fn1 = os.path.join(config.LIGHTCURVE_DIR, f'Validation_Results_Roll_1.hdf5')
        
        save_to_hdf5(save_fn0, validation_results0)
        save_to_hdf5(save_fn1, validation_results1)
    
    # Combine and analyze light curves
    samples, best_params, ls_params = combine_light_curves(data_roll0, data_roll1)
    
    print("Exoplanet analysis complete!")


if __name__ == "__main__":
    main()