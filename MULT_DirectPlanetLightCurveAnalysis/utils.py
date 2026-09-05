#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for exoplanet light curve analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
from scipy.ndimage import gaussian_filter1d as gf
from sklearn.decomposition import PCA
import gc


def create_directories(directories, verbose):
    """
    Create directories if they don't exist.
    
    Parameters
    ----------
    directories : list
        List of directory paths to create
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok = True)
            if verbose:
                print(f"Created directory: {directory}")


def calculate_planet_pixels(sep, pa, roll, pixel_scale, sep_err=None, pa_err=None, roll_err=None, plot=False, 
                           star_x_center=159.5, star_y_center=159.5):
    """
    Calculate the planet pixel coordinates from separation and position angle measurements.

    Parameters
    ----------
    sep : float
        Planet separation in milliarcseconds (mas)
    pa : float
        Planet position angle in degrees (measured E of N)
    roll : float
        Telescope roll angle in degrees
    pixel_scale : float
        Detector pixel scale in arcseconds/pixel
    sep_err : float, optional
        Uncertainty in separation measurement (mas)
    pa_err : float, optional
        Uncertainty in position angle measurement (degrees)
    roll_err : float, optional
        Uncertainty in roll angle (degrees)
    plot : bool, optional
        If True, generate a plot of the Monte Carlo simulation
    star_x_center : float, optional
        X coordinate of the star center
    star_y_center : float, optional
        Y coordinate of the star center

    Returns
    -------
    tuple or dict
        If no uncertainties provided: (x_planet, y_planet) coordinates
        If uncertainties provided: dictionary with keys:
            - 'x': planet x coordinate
            - 'y': planet y coordinate
            - 'x_err': uncertainty in x coordinate
            - 'y_err': uncertainty in y coordinate
            - 'xy_cov': covariance between x and y (if using Monte Carlo)

    Notes
    -----
    - The star is assumed to be at the specified center coordinates
    - Position angle is measured East of North in degrees
    - Coordinates follow the FITS convention: (0,0) at bottom-left
    - Monte Carlo error propagation is used when uncertainties are provided
    """
    # Star position
    xc, yc = star_x_center, star_y_center

    # If uncertainties are provided, use Monte Carlo error propagation
    if any(err is not None for err in [sep_err, pa_err, roll_err]):
        # Set default values for missing uncertainties
        sep_err = 0.0 if sep_err is None else sep_err
        pa_err = 0.0 if pa_err is None else pa_err
        roll_err = 0.0 if roll_err is None else roll_err

        # Number of Monte Carlo samples
        n_samples = 10000

        # Generate random samples
        sep_samples = np.random.normal(sep, sep_err, n_samples)
        pa_samples = np.random.normal(pa, pa_err, n_samples)
        roll_samples = np.random.normal(roll, roll_err, n_samples)

        # Calculate PA in image frame for each sample
        PA_image_samples = pa_samples - roll_samples

        # Convert separation from mas to pixels for each sample
        sep_pixel_samples = sep_samples / (pixel_scale * 1000)

        # Calculate x and y offsets for each sample
        dx_samples = -sep_pixel_samples * np.sin(np.radians(PA_image_samples))
        dy_samples = sep_pixel_samples * np.cos(np.radians(PA_image_samples))

        # Calculate planet positions
        x_planet_samples = xc + dx_samples
        y_planet_samples = yc + dy_samples

        # Calculate means and standard deviations
        x_planet = np.mean(x_planet_samples)
        y_planet = np.mean(y_planet_samples)
        x_err = np.std(x_planet_samples)
        y_err = np.std(y_planet_samples)

        # Calculate covariance
        xy_cov = np.cov(x_planet_samples, y_planet_samples)[0, 1]

        # Show example error ellipse for visualization
        if plot:
            plt.figure(figsize=(8, 6))
            plt.scatter(x_planet_samples, y_planet_samples, s=1, alpha=0.1, color='blue')
            plt.scatter(x_planet, y_planet, color='red', s=50, marker='+')
            plt.scatter(xc, yc, color='yellow', s=100, marker='*')
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')
            plt.title(f'Monte Carlo Simulation of Planet Position\n'
                    f'({x_planet:.2f}±{x_err:.2f}, {y_planet:.2f}±{y_err:.2f})')
            plt.grid(alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            plt.close()
            gc.collect()

        return {
            'x': x_planet,
            'y': y_planet,
            'x_err': x_err,
            'y_err': y_err,
            'xy_cov': xy_cov
        }

    else:
        # Calculate PA in image frame
        PA_image = pa - roll

        # Convert separation from mas to pixels
        sep_pixel = sep / (pixel_scale * 1000)

        # Calculate x and y offsets
        dx = -sep_pixel * np.sin(np.radians(PA_image))
        dy = sep_pixel * np.cos(np.radians(PA_image))

        # Calculate planet position
        x_planet = xc + dx
        y_planet = yc + dy

        return x_planet, y_planet


def blink_images(image1, image2, centroid1, centroid2, interval=200, title1="Image 1", title2="Image 2"):
    """
    Create an interactive display that blinks between two images.

    Parameters:
    -----------
    image1, image2 : numpy.ndarray
        The two images to blink between
    centroid1, centroid2 : tuple
        The (x, y) coordinates of the centroids to highlight in each image
    interval : int, optional
        Time interval between images in milliseconds (default: 500)
    title1, title2 : str, optional
        Titles for the respective images

    Returns:
    --------
    ani : matplotlib.animation.FuncAnimation
        The animation object
    """
    import matplotlib.animation as animation
    from matplotlib.widgets import Button

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))    
    vmin = min(np.percentile(image1, 1), np.percentile(image2, 1))
    vmax = max(np.percentile(image1, 99.99), np.percentile(image2, 99.99))
    plt.subplots_adjust(bottom=0.2)  # Make room for buttons
    plt.xlim([110, 210])
    plt.ylim([110, 210])

    # Initialize with the first image
    img = ax.imshow(image1, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    ax.scatter(159.5, 159.5, s=120, marker='*', facecolor='yellow', edgecolor='black')  # Mark the star center
    title_obj = ax.set_title(title1)
    # Create scatter plots for the markers (initially empty)
    scatter1 = ax.scatter([], [], s=100, marker='o', facecolor='none', edgecolor='red')  # For image1

    # Animation update function
    is_image1 = [True]  # Using list to make it mutable in nested functions

    def update(frame):
        if is_image1[0]:
            img.set_data(image2)
            title_obj.set_text(title2)
            is_image1[0] = False
            scatter1.set_offsets((centroid2[0], centroid2[1]))  # Update the position of the scatter plot
        else:
            img.set_data(image1)
            title_obj.set_text(title1)
            is_image1[0] = True
            scatter1.set_offsets((centroid1[0], centroid1[1]))  # Update the position of the scatter plot
        return [img]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=2, interval=interval, blit=True, repeat=True
    )

    # Add control buttons
    ax_faster = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_slower = plt.axes([0.81, 0.05, 0.1, 0.075])
    ax_toggle = plt.axes([0.25, 0.05, 0.2, 0.075])
    
    plt.close('all')
    gc.collect()

    btn_faster = Button(ax_faster, 'Faster')
    btn_slower = Button(ax_slower, 'Slower')
    btn_toggle = Button(ax_toggle, 'Pause/Resume')

    # Animation state
    running = True

    # Button callbacks
    def faster(event):
        nonlocal interval
        interval = max(50, interval - 100)
        ani.event_source.interval = interval

    def slower(event):
        nonlocal interval
        interval += 100
        ani.event_source.interval = interval

    def toggle(event):
        nonlocal running
        if running:
            ani.event_source.stop()
            running = False
        else:
            ani.event_source.start()
            running = True

    btn_faster.on_clicked(faster)
    btn_slower.on_clicked(slower)
    btn_toggle.on_clicked(toggle)    
    return ani


def pca_timeseries(X, n_components=None):
    """
    Perform PCA on time series data using sklearn.

    Args:
        X: Input array of shape (N, I) where N is number of time points, I is number of series
        n_components: Number of principal components to return. If None, returns all.

    Returns:
        Principal components of shape (N, C) where C is number of components
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    # Print variance explained
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.1%} variance explained")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.1%}")

    return principal_components


def save_to_hdf5(filename, data_dict):
    """
    Save data dictionary to HDF5 file.
    
    Parameters
    ----------
    filename : str
        HDF5 file path
    data_dict : dict
        Dictionary of data to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with h5py.File(filename, 'w') as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)
    
    print(f"Data saved to {filename}")


def load_from_hdf5(filename):
    """
    Load data from HDF5 file.
    
    Parameters
    ----------
    filename : str
        HDF5 file path
        
    Returns
    -------
    dict
        Dictionary of loaded data
    """
    data_dict = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            data_dict[key] = f[key][()]
    
    return data_dict