#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for analyzing time-series images of exoplanets.

This script extracts light curves, removes systematic noise, and identifies signals
from time-series images of an exoplanet.
"""

import os
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d as gf
import spaceKLIP
import sys
import gc

# Resolve local imports from this file, independent of the current working directory.
FilterName = os.environ.get('JWST_FILTER', 'F410M').upper()

project_dir = Path(__file__).resolve().parents[1]
project_parent = project_dir.parent
sys.path.insert(0, str(project_parent))

if FilterName == 'F410M':
    import psfInjectionConfig_F410M as config
elif FilterName == 'F210M':
    import psfInjectionConfig_F210M as config
else:
    raise ValueError("JWST_FILTER must be F210M or F410M")
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.utils import (create_directories, calculate_planet_pixels, 
                  pca_timeseries, save_to_hdf5, load_from_hdf5)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.aperture_photometry import (extract_aperture_photometry, calculate_planet_contribution)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.visualization import (plot_aperture_positions, visualize_all_apertures, plot_raw_lightcurves, 
                         plot_systematic_models, plot_pca_components, plot_stellar_model)
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.BetaPicStellarLCModel import BetaPicStellarLC



if config.USE_MCMC:
    from BetaPic.MULT_DirectPlanetLightCurveAnalysis.modeling import (fit_lightcurve, visualize_results, fit_fourier, visualize_fourier_fit)
    from BetaPic.MULT_DirectPlanetLightCurveAnalysis.validation import (calculate_validation_positions, process_validation_apertures,
                      validate_systematic_correction, visualize_all_apertures)
else:
    from BetaPic.MULT_DirectPlanetLightCurveAnalysis.modeling_nomcmc import (fit_lightcurve, visualize_results, fit_fourier, visualize_fourier_fit)
    from BetaPic.MULT_DirectPlanetLightCurveAnalysis.validation import (calculate_validation_positions, process_validation_apertures,
                      validate_systematic_correction, visualize_all_apertures)


def setup_directories(processing_config):
    """
    Create all directories needed for one injection/recovery run.
    """

    output_root = Path(processing_config["output_data_root"])
    output_dir = Path(processing_config["output_data_dir"])
    planet_dir = Path(processing_config["planet_contribution_dir"])

    lightcurve_dir = output_root / "lightcurves"
    plot_dir = output_root / "plots"

    dirs = [
        lightcurve_dir,
        plot_dir,
        output_dir,
        planet_dir,
        plot_dir / "apertures",
        plot_dir / "lightcurves",
        plot_dir / "pca",
        plot_dir / "modeling",
        plot_dir / "validation",
    ]

    create_directories(dirs, verbose=False)

    return {
        "output_root": output_root,
        "lightcurve_dir": lightcurve_dir,
        "plot_dir": plot_dir,
    }


def initialize_database(processing_config):
    """Initialize the spaceKLIP database for this simulation."""

    database = spaceKLIP.database.create_database(
        input_dir=processing_config["output_data_dir"],
        file_type="calints.fits",
        output_dir=processing_config["output_data_root"],
        pid=processing_config["pid"],
        verbose=False,
    )
    key = processing_config["observation_key"]
    
    if config.VERBOSE:
        print("Database Summary:")
        print("Available keys:", database.obs[key].keys())

    return database, key


def calculate_planet_positions(database, key, processing_config):

    pixel_scale = database.obs[key]["PIXSCALE"].value[0]

    sep = processing_config["separation"]
    pa = processing_config["pa"]

    print(f"Separation: {sep:.2f} mas")
    print(f"PA: {pa:.2f} deg")

    roll1_list = database.obs[key]["ROLL_REF"].value[0:10]
    roll2_list = database.obs[key]["ROLL_REF"].value[10:20]

    roll1 = np.mean(roll1_list)
    roll1_err = np.std(roll1_list)

    roll2 = np.mean(roll2_list)
    roll2_err = np.std(roll2_list)

    print(f"Roll1: {roll1:.5f} ± {roll1_err:.5f}")
    print(f"Roll2: {roll2:.5f} ± {roll2_err:.5f}")

    planet1_pos = calculate_planet_pixels(
        sep,
        pa,
        roll1,
        pixel_scale,
        sep_err=0,
        pa_err=0,
        roll_err=roll1_err,
        star_x_center=config.STAR_X_CENTER,
        star_y_center=config.STAR_Y_CENTER,
    )

    planet2_pos = calculate_planet_pixels(
        sep,
        pa,
        roll2,
        pixel_scale,
        sep_err=0,
        pa_err=0,
        roll_err=roll2_err,
        star_x_center=config.STAR_X_CENTER,
        star_y_center=config.STAR_Y_CENTER,
    )

    print("\nPlanet Pixel Coordinates:")
    print(
        f"Roll 1: "
        f"({planet1_pos['x']:.3f} ± {planet1_pos['x_err']:.3f}, "
        f"{planet1_pos['y']:.3f} ± {planet1_pos['y_err']:.3f})"
    )

    print(
        f"Roll 2: "
        f"({planet2_pos['x']:.3f} ± {planet2_pos['x_err']:.3f}, "
        f"{planet2_pos['y']:.3f} ± {planet2_pos['y_err']:.3f})"
    )

    return (
        (planet1_pos["x"], planet1_pos["y"]),
        (planet2_pos["x"], planet2_pos["y"]),
        roll1,
        roll2,
    )

def calculate_comparison_positions(
    roll1,
    roll2,
    pixel_scale,
    processing_config,
):
    """Calculate positions for comparison apertures."""

    pa_list = processing_config["comparison_pa_list"]
    separation_list = processing_config["comparison_separation_list"]

    xc_comp_list1 = np.zeros(len(pa_list))
    yc_comp_list1 = np.zeros(len(pa_list))
    xc_comp_list2 = np.zeros(len(pa_list))
    yc_comp_list2 = np.zeros(len(pa_list))

    for i, pa in enumerate(pa_list):

        sep = separation_list[i]

        planet_pos1 = calculate_planet_pixels(
            sep,
            pa,
            roll1,
            pixel_scale,
            star_x_center=config.STAR_X_CENTER,
            star_y_center=config.STAR_Y_CENTER,
        )

        planet_pos2 = calculate_planet_pixels(
            sep,
            pa,
            roll2,
            pixel_scale,
            star_x_center=config.STAR_X_CENTER,
            star_y_center=config.STAR_Y_CENTER,
        )

        xc_comp_list1[i] = planet_pos1[0]
        yc_comp_list1[i] = planet_pos1[1]

        xc_comp_list2[i] = planet_pos2[0]
        yc_comp_list2[i] = planet_pos2[1]

    return (
        (xc_comp_list1, yc_comp_list1),
        (xc_comp_list2, yc_comp_list2),
    )

def process_roll(database, key, roll_idx, planet_pos, comp_pos, aperture_radius, processing_config):
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
        processing_config["planet_contribution_dir"], 
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
        processing_config["lightcurve_dir"], 
        f'Planet_LightCurves_roll_{roll_idx}_aper_{aperture_radius:.1f}.hdf5'
    )
    save_to_hdf5(save_filename, result_data)
    
    # Plot raw light curves
    plot_raw_lightcurves(
        data['timeList'], data['fluxList'], data['fluxErrList'], 
        planet_contribution=planet_contribution,
        output_path=os.path.join(
            processing_config["plot_dir"], 'lightcurves', 
            f'{config.FILTER}_roll{roll_idx}_raw_lightcurves.png'
        )
    )
    
    return result_data


def remove_systematics(data, roll_idx, processing_config, stellar_model=None):
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
                processing_config["plot_dir"], 'modeling', 
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
            processing_config["plot_dir"], 'modeling', 
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
            processing_config["plot_dir"], 'pca', 
            f'{config.FILTER}_roll{roll_idx}_pca_components.png'
        )
    )
    gc.collect()
    
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
    
    if config.USE_MCMC:
    
        # Fit the light curve
        samples, best_params = fit_lightcurve(
            t_hours, lc, stellar_model, PC1, PC2, PC3,
            yerr=lc_err, priors=config.LC_PRIORS,
            nwalkers=config.MCMC_WALKERS, nsteps=config.MCMC_STEPS, discard=config.MCMC_BURNIN,
            fit_planet_factor=False, planet_factor_fixed=planet_factor
        )
        
        # Visualize the fit
        visualize_results(
            t_hours, lc, lc_err, best_params, stellar_model, PC1, PC2, PC3,
            samples=samples,
            output_path=os.path.join(
                processing_config["plot_dir"], 'modeling', 
                f'{config.FILTER}_roll{roll_idx}_lightcurve_fit_planet.png'
            )
        )

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
                fit_planet_factor=False, planet_factor_fixed=planet_factor_comp
            )
            
            # Visualize the fit
            visualize_results(
                t_hours, lc_comp, lc_err_comp, best_params_comp, stellar_model, PC1, PC2, PC3,
                samples=samples_comp,
                output_path=os.path.join(
                    processing_config["plot_dir"], 'modeling', 
                    f'{config.FILTER}_roll{roll_idx}_lightcurve_fit_aperture_{aperture_i}.png'
                )
            )
            
            del samples_comp
            del best_params_comp

    else:
        # Fit the light curve
        best_params = fit_lightcurve(
            t_hours, lc, stellar_model, PC1, PC2, PC3,
            yerr=lc_err, priors=config.LC_PRIORS,
            fit_planet_factor=False, planet_factor_fixed=planet_factor
        )
        
        # Visualize the fit
        visualize_results(
            t_hours, lc, lc_err, best_params, stellar_model, PC1, PC2, PC3,
            output_path=os.path.join(
                processing_config["plot_dir"], 'modeling', 
                f'{config.FILTER}_roll{roll_idx}_lightcurve_fit_planet.png'
            )
        )

        # perform the same fit to the comparison apertures
        for aperture_i in range(1, fluxList.shape[1]):
            planet_factor_comp = planet_contribution[aperture_i]        
            print(f'Planet contribution in aperture {aperture_i}: {planet_factor_comp:.4f}')
            
            # Prepare data for fitting
            lc_comp = fluxList[:, aperture_i] / np.median(fluxList[:, aperture_i])
            lc_err_comp = fluxErrList[:, aperture_i] / np.median(fluxList[:, aperture_i])
            
            # Fit the light curve
            best_params_comp = fit_lightcurve(
                t_hours, lc_comp, stellar_model, PC1, PC2, PC3,
                yerr=lc_err_comp, priors=config.LC_PRIORS,
                fit_planet_factor=False, planet_factor_fixed=planet_factor_comp
            )
            
            # Visualize the fit
            visualize_results(
                t_hours, lc_comp, lc_err_comp, best_params_comp, stellar_model, PC1, PC2, PC3,
                output_path=os.path.join(
                    processing_config["plot_dir"], 'modeling', 
                    f'{config.FILTER}_roll{roll_idx}_lightcurve_fit_aperture_{aperture_i}.png'
                )
            )  
    
    # Print best-fit parameters
    print("\nBest-fit parameters:")
    param_names = ['A', 'B', 'P', 'w0', 'w1', 'w2', 'w3']
    for name, val in zip(param_names, best_params):
        print(f"{name}: {val:.6f}")
    
    # Extract models
    systematic_model = best_params[3] + best_params[4] * PC1 + best_params[5] * PC2 + best_params[6] * PC3
    stellar_lc_factor = (1 - planet_factor) * stellar_model
    astrophysical_model = stellar_lc_factor + planet_factor * (
        1 + best_params[0] * np.sin(2*np.pi*t_hours/best_params[2]) + 
        best_params[1] * np.cos(2*np.pi*t_hours/best_params[2])
    )
    
    # Calculate planet light curve corrected for systematics
    planet_lc_corrected = (lc / systematic_model - stellar_lc_factor) / planet_factor
    
    # Save corrected light curve
    save_fn = os.path.join(
        processing_config["lightcurve_dir"], 
        f'Planet_LightCurve_Corrected_Roll_{roll_idx}.txt'
    )
    np.savetxt(save_fn, np.array([timeList, planet_lc_corrected]).T)
    
    return planet_lc_corrected, systematic_model, astrophysical_model, systematic_PCs


def combine_light_curves(processing_config):
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
    fn_roll0 = os.path.join(processing_config["lightcurve_dir"], f'Planet_LightCurve_Corrected_Roll_0.txt')
    fn_roll1 = os.path.join(processing_config["lightcurve_dir"], f'Planet_LightCurve_Corrected_Roll_1.txt')
    
    t1, lc1 = np.loadtxt(fn_roll0, unpack=True)
    t2, lc2 = np.loadtxt(fn_roll1, unpack=True)
    
    # Combine data
    t = np.concatenate([t1, t2])
    t0 = t1[0]  # Reference time
    t_hours = (t - t0) * 24  # Convert to hours
    lc = np.concatenate([lc1, lc2])
    
    if config.USE_MCMC:
        # Fit Fourier model
        samples, best_params, ls_params = fit_fourier(
            t_hours, lc, priors=config.FOURIER_PRIORS,
            nwalkers=config.MCMC_WALKERS, nsteps=config.MCMC_STEPS, discard=config.MCMC_BURNIN
        )
        
        # Visualize fit
        inject_params = np.array([processing_config["amplitude"] * processing_config["phase"],
                                    processing_config["amplitude"] * processing_config["phase"], 
                                    processing_config["period"]])
        visualize_fourier_fit(
            t_hours, lc, samples, best_params, ls_params, inject_params=inject_params,
            output_path=os.path.join(
                processing_config["plot_dir"], 'modeling', 
                f'{config.FILTER}_combined_fourier_fit.png'
            )
        )
        
        del t1, t2, lc1, lc2, t, t_hours, lc
        gc.collect()
        
        return samples, best_params, ls_params
    
    else:
        # Fit Fourier model
        ls_params = fit_fourier(t_hours, lc, priors=config.FOURIER_PRIORS)
        
        # Visualize fit
        inject_params = np.array([processing_config["amplitude"] * processing_config["phase"],
                                    processing_config["amplitude"] * processing_config["phase"], 
                                    processing_config["period"]])
        visualize_fourier_fit(
            t_hours, lc, ls_params, inject_params=inject_params,
            output_path=os.path.join(
                processing_config["plot_dir"], 'modeling', 
                f'{config.FILTER}_combined_fourier_fit.png'
            )
        )
        
        del t1, t2, lc1, lc2, t, t_hours, lc
        gc.collect()
        
        return ls_params 

def run_analysis(processing_config):
    
    database, key = initialize_database(processing_config)
    paths = setup_directories(processing_config)

    pixel_scale = database.obs[key]['PIXSCALE'].value[0]
    planet_pos, planet_pos2, roll1, roll2 = \
        calculate_planet_positions(
            database,
            key,
            processing_config,
        )

    comp_pos = calculate_comparison_positions(
        roll1,
        roll2,
        pixel_scale,
        processing_config,
    )

    # Validation positions
    if config.ENABLE_VALIDATION:
        validation_pos = calculate_validation_positions(
            roll1,
            roll2,
            pixel_scale,
            config.VALIDATION_PA_LIST,
            config.VALIDATION_SEPARATION_LIST,
            config.STAR_X_CENTER,
            config.STAR_Y_CENTER,
        )

    else:
        validation_pos = None

    # Process rolls
    data_roll0 = process_roll(
        database,
        key,
        0,
        (planet_pos, planet_pos2),
        comp_pos,
        config.APERTURE_RADIUS,
        processing_config,
    )

    data_roll1 = process_roll(
        database,
        key,
        1,
        (planet_pos, planet_pos2),
        comp_pos,
        config.APERTURE_RADIUS,
        processing_config,
    )
    
    # Load stellar models
    stellar_lc = BetaPicStellarLC(
        data_roll0['timeList']
    )

    if processing_config["filter"] == "F410M":
        stellar_model0 = stellar_lc.F410M_LC
    elif processing_config["filter"] == "F210M":
        stellar_model0 = stellar_lc.F210M_LC
    else:
        raise ValueError(
            f"Unsupported filter: "
            f"{processing_config['filter']}"
        )

    stellar_lc = BetaPicStellarLC(
        data_roll1['timeList']
    )

    if processing_config["filter"] == "F410M":
        stellar_model1 = stellar_lc.F410M_LC
    elif processing_config["filter"] == "F210M":
        stellar_model1 = stellar_lc.F210M_LC
    
    # Remove systematics
    planet_lc0, systematic0, astro0, pcs0 = \
        remove_systematics(
            data_roll0,
            0,
            processing_config,
            stellar_model=stellar_model0,
        )

    planet_lc1, systematic1, astro1, pcs1 = \
        remove_systematics(
            data_roll1,
            1,
            processing_config,
            stellar_model=stellar_model1,
        )

    # Combine rolls
    if config.USE_MCMC:
        samples, best_params, ls_params = combine_light_curves(processing_config)
    else:
        ls_params = combine_light_curves(processing_config)
    
    del data_roll0
    del data_roll1
    del systematic0
    del systematic1
    del astro0
    del astro1
    del pcs0
    del pcs1
    gc.collect()
    
    lc = np.concatenate([
        planet_lc0,
        planet_lc1,
    ])
    
    # Return everything useful
    completion_file = Path(processing_config["output_data_root"]) / "analysis_complete.txt"
    completion_file.write_text("Analysis completed successfully.\n")


    if config.USE_MCMC:
        return {
            "lightcurve": lc,

            "best_params": best_params,
            "ls_params": ls_params,

            "planet_lc_roll0": planet_lc0,
            "planet_lc_roll1": planet_lc1,

            "processing_config": processing_config,
        }
    else:
        return {
            "lightcurve": lc,

            "ls_params": ls_params,

            "planet_lc_roll0": planet_lc0,
            "planet_lc_roll1": planet_lc1,

            "processing_config": processing_config,
        }        