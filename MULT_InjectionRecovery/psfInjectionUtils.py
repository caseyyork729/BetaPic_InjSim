#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for PSF injection simulation.
"""

import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
import glob
from concurrent.futures import ProcessPoolExecutor
import warnings
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from pathlib import Path
from photutils.aperture import CircularAperture, aperture_photometry

def calculate_psf_position(separation_mas, pa_deg, roll_deg, pixel_scale, star_x, star_y):
    """
    Calculate PSF injection position in pixels.
    
    Parameters
    ----------
    separation_mas : float
        Separation in milliarcseconds
    pa_deg : float
        Position angle in degrees (North=0, East=90)
    roll_deg : float
        Telescope roll angle in degrees
    pixel_scale : float
        Pixel scale in arcsec/pixel
    star_x, star_y : float
        Star center coordinates in pixels
    
    Returns
    -------
    tuple
        (x, y) position in pixels
    """
    # Convert separation to arcseconds
    sep_arcsec = separation_mas / 1000.0
    
    # Calculate offset PA (instrument PA = sky PA - roll angle)
    pa_inst = pa_deg - roll_deg
    
    # Convert to radians
    pa_rad = np.deg2rad(pa_inst)
    
    # Calculate pixel offsets (North=+Y, East=+X in detector coordinates)
    dx_arcsec = -sep_arcsec * np.sin(pa_rad)
    dy_arcsec = sep_arcsec * np.cos(pa_rad)
    
    # Convert to pixels
    dx_pix = dx_arcsec / pixel_scale
    dy_pix = dy_arcsec / pixel_scale
    
    # Calculate final position
    x_pos = star_x + dx_pix
    y_pos = star_y + dy_pix
    
    return x_pos, y_pos

def load_psf_model(psf_path):
    """
    Load PSF model from FITS file.
    
    Parameters
    ----------
    psf_path : str
        Path to PSF FITS file
    
    Returns
    -------
    ndarray
        PSF model array
    """
    if not os.path.exists(psf_path):
        raise FileNotFoundError(f"PSF file not found: {psf_path}")
    
    with fits.open(psf_path) as hdul:
        psf_model = hdul[0].data
        if psf_model.ndim == 3:
            psf_model = psf_model[0]  # Take first frame if 3D
    
    return psf_model

def calculate_flux_modulation(t0, time_array, period_hours, amplitude, phase=0.0, base_flux=1.0):
    """
    Calculate sinusoidal flux modulation.
    
    Parameters
    ----------
    time_array : array
        Time array in MJD
    period_hours : float
        Period in hours
    amplitude : float
        Amplitude of modulation (fraction)
    phase : float
        Phase offset in radians
    base_flux : float
        Base flux level
    
    Returns
    -------
    array
        Flux modulation factors
    """
    # Convert time to hours from start    
    t_hours = (time_array - t0) * 24
    
    # Calculate sinusoidal modulation
    flux_factors = base_flux + amplitude * np.sin(2 * np.pi * t_hours / period_hours + phase)
    
    return flux_factors

def inject_psf_single_frame(data_frame, psf_model, x_pos, y_pos, flux_factor, x_pos_b=None, y_pos_b=None, flux_b=0.0, dxy=0):
    """
    Inject PSF into a single frame using precise subpixel positioning.
    
    Parameters
    ----------
    data_frame : ndarray
        2D data frame
    psf_model : ndarray
        PSF model
    x_pos, y_pos : float
        Injection position (precise coordinates)
    flux_factor : float
        Flux scaling factor
    
    Returns
    -------
    ndarray
        Frame with injected PSF
    """
    from scipy.ndimage import shift
    
    # Create copy of data
    injected_frame = data_frame.copy()
    
    # Get dimensions
    ny, nx = data_frame.shape
    ny_psf, nx_psf = psf_model.shape
    
    # Expand the PSF to the size of the science image
    psf_expanded = np.zeros((ny, nx))
    psf_expanded[:ny_psf, :nx_psf] = psf_model
    
    # Calculate centers
    xc = nx / 2 - 0.5  # Center of science image
    yc = ny / 2 - 0.5
    xc_psf = nx_psf / 2 - 0.5  # Center of PSF
    yc_psf = ny_psf / 2 - 0.5
    
    # Calculate shift needed to place PSF at target position, incorporating random noise in injection
    pointing_error_x = dxy * np.random.randn()
    pointing_error_y = dxy * np.random.randn()
    x_shift = x_pos - xc_psf + pointing_error_x
    y_shift = y_pos - yc_psf + pointing_error_y
    
    # Apply subpixel shift using scipy.ndimage.shift
    yxshift = np.array([y_shift, x_shift])
    psf_shifted = shift(psf_expanded, yxshift, order=3)
    injected_psf = psf_shifted * flux_factor

    # remove Beta Pic b contribution if specified
    if x_pos_b is not None and y_pos_b is not None and flux_b > 0:
        
        # Calculate shift for Beta Pic b
        x_shift_b = x_pos_b - xc_psf
        y_shift_b = y_pos_b - yc_psf
        
        # Apply subpixel shift for Beta Pic b
        yxshift_b = np.array([y_shift_b, x_shift_b])
        psf_shifted_b = flux_b * shift(psf_expanded, yxshift_b, order=3)
        
        # Subtract Beta Pic b contribution
        psf_shifted -= psf_shifted_b
    
    # Add shifted PSF to the frame
    injected_frame += injected_psf
    
    return injected_frame

def calculate_planet_contribution(output_file, psf_model, config, planet_contribution_dir):
    """
    Calculate pixel-by-pixel planet contribution after PSF injection.
    
    Parameters
    ----------
    output_file : str
        Path to the injected FITS file
    psf_model : ndarray
        PSF model used for injection
    config : dict
        Processing configuration
    planet_contribution_dir : str
        Directory to save planet contribution files
    
    Returns
    -------
    str
        Status message
    """
    try:
        # Create output directory
        os.makedirs(planet_contribution_dir, exist_ok=True)
        
        # Open the injected file to get the first frame for contribution calculation
        with fits.open(output_file) as hdul:
            # Use first frame as reference
            data = hdul[1].data[0]
            header = hdul[1].header
            
            # Get image dimensions and parameters
            ny, nx = data.shape
            ny_psf, nx_psf = psf_model.shape
            
            # Get roll angle
            roll_angle = header.get('ROLL_REF', 0.0)
            
            # Calculate pixel scale (similar to reference code)
            pixscale = config['pixel_scale']
            
            # Expand PSF to science image size
            psf_expanded = np.zeros((ny, nx))
            psf_expanded[:ny_psf, :nx_psf] = psf_model
            
            # Calculate planet position
            PA_image = config['pa'] - roll_angle
            separation_px = config['separation'] / pixscale / 1000
            dx_sep = -separation_px * np.sin(np.deg2rad(PA_image))
            dy_sep = separation_px * np.cos(np.deg2rad(PA_image))
            
            # Calculate centers
            xc = nx / 2 - 0.5
            yc = ny / 2 - 0.5
            xc_psf = nx_psf / 2 - 0.5
            yc_psf = ny_psf / 2 - 0.5
            xc_planet = xc + dx_sep
            yc_planet = yc + dy_sep
            
            # Calculate shift
            x_shift = xc_planet - xc_psf
            y_shift = yc_planet - yc_psf
            yxshift = np.array([y_shift, x_shift])
            
            # Create shifted PSF with base flux
            psf_shifted = config['flux'] * shift(psf_expanded, yxshift, order=3)
            
            # Calculate planet contribution (ratio of injected PSF to science data)
            # Avoid division by zero
            data_safe = np.where(data != 0, data, np.nan)
            planet_contribution = psf_shifted / data_safe
            
            # Create output filenames
            basename = os.path.basename(output_file)
            contribution_fn = basename.replace('.fits', '_planet_contribution.fits')
            psf_model_fn = basename.replace('.fits', '_psf_model.fits')
            plot_fn = basename.replace('.fits', '_planet_contribution.png')
            
            contribution_path = os.path.join(planet_contribution_dir, contribution_fn)
            psf_model_path = os.path.join(planet_contribution_dir, psf_model_fn)
            plot_path = os.path.join(planet_contribution_dir, plot_fn)
            
            # Save planet contribution
            fits.writeto(contribution_path, planet_contribution, overwrite=True)
            
            # Save PSF model
            fits.writeto(psf_model_path, psf_shifted, overwrite=True)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            vmin = np.nanpercentile(data, 0.5)
            vmax = np.nanpercentile(data, 99.95)
            
            ax.imshow(data, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
            ax.plot(xc_planet, yc_planet, 'ro', markersize=8, mfc='none', mew=1.5, label='Planet')
            ax.plot(xc, yc, 'bx', markersize=8, mew=2, label='Star center')
            
            ax.set_title(f'Planet Injection\nRoll: {roll_angle:.2f}°, Sep: {separation_px:.1f} px\nPA: {config["pa"]:.1f}°, Flux: {config["flux"]:.0f}')
            ax.legend()
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return f"Planet contribution calculated for: {basename}"
            
    except Exception as e:
        return f"Error calculating planet contribution for {os.path.basename(output_file)}: {str(e)}"

def process_single_file(file_info):
    """
    Process a single FITS file for PSF injection.
    
    Parameters
    ----------
    file_info : tuple
        (input_path, output_path, psf_model, config_dict, planet_contribution_dir)
    
    Returns
    -------
    str
        Status message
    """
    input_path, output_path, psf_model, config, planet_contribution_dir = file_info
    
    try:
        # Open input file
        with fits.open(input_path) as hdul:
            # Copy HDU list
            new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
            
            # Get data cube and timing information
            data_cube = new_hdul[1].data  # Science data
            header = new_hdul[1].header
            
            # Extract timing information from extension 5
            try:
                time_data = new_hdul[5].data
                time_array = time_data['int_mid_BJD_TDB']  # BJD times
            except (IndexError, KeyError):
                # Fallback: use header timing information
                if 'MJD-BEG' in header and 'MJD-END' in header:
                    mjd_start = header['MJD-BEG']
                    mjd_end = header['MJD-END']
                    n_frames = data_cube.shape[0]
                    time_array = np.linspace(mjd_start, mjd_end, n_frames)
                else:
                    # Last resort: create dummy time array
                    time_array = np.arange(data_cube.shape[0]) / 24.0
                    warnings.warn(f"Could not extract timing data from {os.path.basename(input_path)}, using dummy times")
            
            # Get roll angle
            roll_angle = header.get('ROLL_REF', 0.0)
            
            # Calculate PSF position
            x_pos, y_pos = calculate_psf_position(
                config['separation'], config['pa'], roll_angle, 
                config['pixel_scale'], config['star_x'], config['star_y']
            )

            # Calculate the postion of Beta Pic b and remove it from the injection frame
            x_pos_b, y_pos_b = calculate_psf_position(
                config['sep_b_pic_b'], config['pa_b_pic_b'], roll_angle, 
                config['pixel_scale'], config['star_x'], config['star_y']
            )
            flux_b = config['flux_b_pic_b']
            
            # Calculate flux modulation using proper time array
            flux_factors = calculate_flux_modulation(
                config['t0'], time_array, config['period'], config['amplitude'], 
                config['phase'], config['flux']
            )

            for i in range(data_cube.shape[0]):
                data_cube[i] = inject_psf_single_frame(
                    data_cube[i],
                    psf_model,
                    x_pos,
                    y_pos,
                    flux_factors[i],
                    x_pos_b=x_pos_b,
                    y_pos_b=y_pos_b,
                    flux_b=flux_b,
                    dxy=config["dxy"],
                )

            # Update header with injection metadata
            new_hdul[1].header['PSF_INJ'] = True
            new_hdul[1].header['PSF_X'] = x_pos
            new_hdul[1].header['PSF_Y'] = y_pos
            new_hdul[1].header['PSF_SEP'] = config['separation']
            new_hdul[1].header['PSF_PA'] = config['pa']
            new_hdul[1].header['PSF_ROLL'] = roll_angle
            new_hdul[1].header['PSF_PER'] = config['period']
            new_hdul[1].header['PSF_AMP'] = config['amplitude']
            
        # Save output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_hdul.writeto(output_path, overwrite=True)
        new_hdul.close()
        
        # Calculate planet contribution after injection
        contribution_status = calculate_planet_contribution(
            output_path, psf_model, config, planet_contribution_dir
        )
        
        return f"Successfully processed: {os.path.basename(input_path)}\n{contribution_status}"
        
    except Exception as e:
        return f"Error processing {os.path.basename(input_path)}: {str(e)}"

def create_directories(dir_list, verbose):
    """Create directories if they don't exist."""
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)
        if verbose:
            print(f"Created directory: {directory}")