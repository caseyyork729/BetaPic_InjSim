#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization functions for exoplanet light curve analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d as gf
from BetaPic.MULT_DirectPlanetLightCurveAnalysis.misc import rebin
import gc


def plot_aperture_positions(image, planet_position, comparison_positions, 
                          aperture_radius, star_x_center=159.5, star_y_center=159.5,
                          title="Aperture Positions", output_path=None):
    """
    Plot aperture positions on an image.
    
    Parameters
    ----------
    image : array-like
        2D image array
    planet_position : tuple
        (x, y) coordinates of planet aperture
    comparison_positions : list of tuples
        List of (x, y) coordinates for comparison apertures
    aperture_radius : float
        Radius of apertures in pixels
    xlim, ylim : tuple, optional
        Plot limits
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot image
    vmin = np.nanpercentile(image, 1)
    vmax = np.nanpercentile(image, 99.99)
    ax.imshow(image, origin='lower', cmap='plasma', vmin=vmin, vmax=vmax)
    
    # Plot planet aperture
    planet_circle = plt.Circle(planet_position, aperture_radius, color='C0', lw=2, fill=False, zorder=100)
    ax.add_artist(planet_circle)
    
    # Plot comparison apertures
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    for i, (xc, yc) in enumerate(comparison_positions):
        color_idx = min(i, len(colors)-1)
        ax.add_artist(plt.Circle((xc, yc), aperture_radius, color=colors[color_idx], fc='none', lw=2))
    
    # Add star position
    ax.scatter(star_x_center, star_y_center, s=120, marker='*', facecolor='yellow', edgecolor='black')
    
    # Set plot limits and labels
    xlim = (star_x_center - 50, star_x_center + 50)
    ylim = (star_y_center - 50, star_y_center + 50)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    
    plt.tight_layout()
    plt.close("all")
    gc.collect()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_all_apertures(image, planet_position, comparison_positions, validation_positions,
                          aperture_radius, 
                          star_x_center=159.5, star_y_center=159.5,
                          title="All Aperture Positions", output_path=None):
    """
    Plot all aperture positions on an image with different colors for different types.
    
    Parameters
    ----------
    image : array-like
        2D image array
    planet_position : tuple
        (x, y) coordinates of planet aperture
    comparison_positions : list of tuples
        List of (x, y) coordinates for comparison apertures
    validation_positions : list of tuples
        List of (x, y) coordinates for validation apertures
    aperture_radius : float
        Radius of apertures in pixels
    xlim, ylim : tuple, optional
        Plot limits
    star_x_center, star_y_center : float, optional
        Star center coordinates
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot image
    vmin = np.nanpercentile(image, 1)
    vmax = np.nanpercentile(image, 99.99)
    ax.imshow(image, origin='lower', cmap='plasma', vmin=vmin, vmax=vmax)
    
    # Plot star position
    ax.scatter(star_x_center, star_y_center, s=120, marker='*', facecolor='yellow', edgecolor='black', zorder=100, label='Star')
    
    # Plot planet aperture
    planet_circle = plt.Circle(planet_position, aperture_radius, color='C0', lw=2, fill=False, zorder=100, label='Planet')
    ax.add_artist(planet_circle)
    
    # Plot comparison apertures
    comp_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    for i, (xc, yc) in enumerate(comparison_positions):
        color_idx = min(i, len(comp_colors)-1)
        circle = plt.Circle((xc, yc), aperture_radius, color=comp_colors[color_idx], fc='none', lw=2)
        ax.add_artist(circle)
    
    # Add a representative comparison aperture for the legend
    comp_circle = plt.Circle((0, 0), aperture_radius, color='C1', fc='none', lw=2, label='Comparison')
    ax.add_artist(comp_circle)
    comp_circle.set_visible(False)  # Make it invisible, just for legend
    
    # Plot validation apertures
    for i, (xc, yc) in enumerate(validation_positions):
        val_circle = plt.Circle((xc, yc), aperture_radius, color='k', fc='none', lw=2, linestyle='--')
        label_x = xc + aperture_radius * 1.2
        label_y = yc + aperture_radius * 1.2
        ax.text(label_x, label_y, f"{i+1:02d}", color='k', fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        ax.add_artist(val_circle)
    
    # Add a representative validation aperture for the legend
    val_circle = plt.Circle((0, 0), aperture_radius, color='k', fc='none', lw=2, linestyle='--', label='Validation')
    ax.add_artist(val_circle)
    val_circle.set_visible(False)  # Make it invisible, just for legend
    
    
    # Set plot limits and labels
    xlim = (star_x_center - 50, star_x_center + 50)
    ylim = (star_y_center - 50, star_y_center + 50)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.close("all")
    gc.collect()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_raw_lightcurves(timeList, fluxList, fluxErrList, 
                       planet_contribution=None, output_path=None):
    """
    Plot raw light curves for all apertures.
    
    Parameters
    ----------
    timeList : array-like
        Time array
    fluxList : array-like
        Flux measurements for all apertures
    fluxErrList : array-like
        Flux errors for all apertures
    planet_contribution : array-like, optional
        Planet contribution factors
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])
    
    # Plot planet light curve
    norm0 = np.median(fluxList[:, 0])
    fplanet_norm = fluxList[:, 0] / norm0
    ferr_norm = fluxErrList[:, 0] / norm0
    ax0.errorbar(timeList, (fplanet_norm-0.7) / 0.3, yerr=ferr_norm, fmt='.', label='Planet', color='C0')
    
    # Plot comparison light curves
    displacement = 0.05
    for j in range(fluxList.shape[1]):
        normalization = np.median(fluxList[:, j])
        f_norm = fluxList[:, j] / normalization + displacement * j
        ferr_norm = fluxErrList[:, j] / normalization
        ax.errorbar(timeList, f_norm, yerr=ferr_norm, fmt='.', label=f'Aperture {j}')
        
        # Print planet contribution if available
        if planet_contribution is not None and j < len(planet_contribution):
            print(f'In aperture {j}, the planet contribution is {planet_contribution[j]:.4f}')

    ax.legend()
    ax0.set_title('Aperture centered on the planet')
    ax0.set_ylabel('Normalized Flux')
    ax.set_xlabel('BJD')
    ax.set_ylabel('Normalized Flux + Offset')
    ax.set_title('Reference Apertures')
    plt.tight_layout()
    plt.close("all")
    gc.collect()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_systematic_models(timeList, systematicModel, planet_flux=None, 
                          sigma=1, output_path=None):
    """
    Plot systematic models from all comparison apertures.
    
    Parameters
    ----------
    timeList : array-like
        Time array
    systematicModel : array-like
        Systematic models for all apertures
    planet_flux : array-like, optional
        Planet flux for comparison
    sigma : float, optional
        Gaussian filter sigma
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    n_models = systematicModel.shape[1]
    fig, axes = plt.subplots(n_models, 1, figsize=(8, 2*n_models), sharex=True)

    # Handle single model case
    if n_models == 1:
        axes = [axes]

    for i in range(n_models):
        # Plot systematic model
        axes[i].plot(timeList, systematicModel[:, i], marker='o', alpha=0.3, mfc='none', 
                    color=f'C{i}', label=f'Aperture {i+1}')
        axes[i].plot(rebin(timeList, 10), rebin(systematicModel[:, i], 10), 
                     ds='steps-mid',
                     color=f'C{i}')
        
        # Plot planet flux if provided
        if planet_flux is not None:
            norm0 = np.median(planet_flux)
            f0_norm = planet_flux / norm0
            axes[i].plot(timeList, gf(f0_norm, sigma), marker='.', alpha=0.3, 
                        color='k', label='Planet')
            axes[i].plot(rebin(timeList, 10), rebin(f0_norm, 10), ds='steps-mid',
                        color='k')
        
        axes[i].set_ylabel('Normalized Flux')
        axes[i].legend()
        axes[i].set_title(f'Aperture {i+1}')

    axes[-1].set_xlabel('BJD')
    plt.tight_layout()
    plt.close("all")
    gc.collect()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pca_components(timeList, pca_components, n_components=3, output_path=None):
    """
    Plot PCA components of systematic variations.
    
    Parameters
    ----------
    timeList : array-like
        Time array
    pca_components : array-like
        PCA components
    n_components : int, optional
        Number of components to plot
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PCA components
    for i in range(min(n_components, pca_components.shape[1])):
        ax.plot(timeList, pca_components[:, i], label=f'PC{i+1}')
    
    ax.set_xlabel('Time (BJD)')
    ax.set_ylabel('Component Value')
    ax.set_title('PCA Components of Systematic Variations')
    ax.legend()
    
    plt.tight_layout()
    plt.close("all")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_stellar_model(time, stellar_model, comparison_flux=None, 
                     aperture_idx=None, filter_sigma=10, output_path=None):
    """
    Plot stellar model and compare with a comparison aperture.
    
    Parameters
    ----------
    time : array-like
        Time array
    stellar_model : array-like
        Stellar model
    comparison_flux : array-like, optional
        Flux from a comparison aperture
    aperture_idx : int, optional
        Index of comparison aperture
    filter_sigma : float, optional
        Gaussian filter sigma
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot stellar model
    ax.plot(time, stellar_model, label='Stellar model')
    
    # Plot comparison aperture if provided
    if comparison_flux is not None:
        norm = np.median(comparison_flux)
        f_norm = comparison_flux / norm
        ax.plot(time, gf(f_norm, filter_sigma), marker='o', alpha=0.1, mfc='none', 
               label=f'Aperture {aperture_idx}')
        
        # Plot systematic component
        ax.plot(time, gf(f_norm / stellar_model, filter_sigma), label='Systematic model')
    
    ax.set_xlabel('BJD')
    ax.set_ylabel('Normalized flux')
    ax.set_title('Stellar Model Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.close("all")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_validation_results(validation_results, roll_idx, filter_name, output_path=None):
    """
    Plot summary of validation results.
    
    Parameters
    ----------
    validation_results : dict
        Validation results dictionary
    roll_idx : int
        Roll index
    filter_name : str
        Filter name
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    correction_quality = validation_results['correction_quality']
    n_validation = len(correction_quality)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(1, n_validation+1), correction_quality)
    
    # Color bars based on improvement
    for i, bar in enumerate(bars):
        if correction_quality[i] > 1.5:
            bar.set_color('green')
        elif correction_quality[i] > 1.0:
            bar.set_color('lightgreen')
        else:
            bar.set_color('red')
    
    ax.axhline(1.0, color='r', linestyle='--', label='No Improvement')
    ax.set_xlabel('Validation Aperture')
    ax.set_ylabel('Correction Quality (RMS Before / RMS After)')
    ax.set_title(f'Systematic Correction Validation - {filter_name} Roll {roll_idx}')
    ax.set_xticks(range(1, n_validation+1))
    ax.grid(alpha=0.3)
    
    # Add text labels with exact values
    for i, v in enumerate(correction_quality):
        ax.text(i+1, v+0.05, f"{v:.2f}x", ha='center')
    
    ax.legend()
    
    plt.tight_layout()
    plt.close("all")
    gc.collect()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig