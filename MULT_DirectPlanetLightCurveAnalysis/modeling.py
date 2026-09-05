#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Light curve modeling and systematic noise removal.
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import corner
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d as gf
import gc

from BetaPic.MULT_DirectPlanetLightCurveAnalysis.misc import rebin, rebin_std

def lc_model(t, params, stellarmodel, PC1, PC2, PC3, fit_planet_factor=False, planet_factor_fixed=0.):
    """
    Light curve model combining astrophysical and systematic components.

    Parameters
    ----------
    t : array-like
        Time array
    params : list or array
        Model parameters. If fit_planet_factor=False: [A, B, P, w0, w1, w2, w3, planet_factor_fixed]
        If fit_planet_factor=True: [A, B, P, w0, w1, w2, w3, planet_factor]
    stellarmodel : array-like
        Stellar light curve model
    PC1, PC2, PC3 : array-like
        Principal components of systematic variations
    fit_planet_factor : bool, optional
        Whether planet_factor is a free parameter

    Returns
    -------
    array-like
        Model light curve
    """
    if fit_planet_factor:
        A, B, P, w0, w1, w2, w3, planet_factor = params
    else:
        A, B, P, w0, w1, w2, w3 = params[:7]
        planet_factor = planet_factor_fixed

    # Planet model
    stellar_factor = 1.0 - planet_factor
    planetmodel = 1 + A * np.sin(2*np.pi*t/P) + B * np.cos(2*np.pi*t/P)

    # Systematic model
    systematic_model = w0 + w1*PC1 + w2*PC2 + w3*PC3

    # Combined model
    lc = (stellar_factor * stellarmodel + planet_factor * planetmodel) * systematic_model

    return lc


def chi2_func(params, t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr, fit_planet_factor=False, planet_factor_fixed=0.):
    """
    Chi-squared function for least squares fitting.
    
    Parameters
    ----------
    params : list or array
        Model parameters
    t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr : array-like
        Model inputs and observed data
    fit_planet_factor : bool, optional
        Whether planet_factor is a free parameter
        
    Returns
    -------
    float
        Chi-squared value
    """
    lc_pred = lc_model(t, params, stellarmodel, PC1, PC2, PC3, fit_planet_factor, planet_factor_fixed)
    chi2 = np.sum((lc_obs - lc_pred)**2 / yerr**2)
    return chi2


def log_likelihood(params, t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr, fit_planet_factor=False, planet_factor_fixed=0.):
    """
    Log likelihood function for MCMC.
    
    Parameters
    ----------
    params : list or array
        Model parameters
    t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr : array-like
        Model inputs and observed data
    fit_planet_factor : bool, optional
        Whether planet_factor is a free parameter
        
    Returns
    -------
    float
        Log likelihood value
    """
    lc_pred = lc_model(t, params, stellarmodel, PC1, PC2, PC3, fit_planet_factor, planet_factor_fixed)
    chi2 = np.sum((lc_obs - lc_pred)**2 / yerr**2)
    return -0.5 * chi2


def log_prior(params, priors, fit_planet_factor=False):
    """
    Log prior function for MCMC.
    
    Parameters
    ----------
    params : list or array
        Model parameters
    priors : dict
        Dictionary with parameter bounds
    fit_planet_factor : bool, optional
        Whether planet_factor is a free parameter
        
    Returns
    -------
    float
        Log prior value (0 or -infinity)
    """
    if fit_planet_factor:
        A, B, P, w0, w1, w2, w3, planet_factor = params
        
        if (priors['A'][0] < A < priors['A'][1] and 
            priors['B'][0] < B < priors['B'][1] and 
            priors['P'][0] < P < priors['P'][1] and 
            priors['w0'][0] < w0 < priors['w0'][1] and
            priors['w1'][0] < w1 < priors['w1'][1] and 
            priors['w2'][0] < w2 < priors['w2'][1] and 
            priors['w3'][0] < w3 < priors['w3'][1] and
            priors['planet_factor'][0] <= planet_factor <= priors['planet_factor'][1]):
            return 0.0
    else:
        A, B, P, w0, w1, w2, w3 = params[:7]
        
        if (priors['A'][0] < A < priors['A'][1] and 
            priors['B'][0] < B < priors['B'][1] and 
            priors['P'][0] < P < priors['P'][1] and 
            priors['w0'][0] < w0 < priors['w0'][1] and
            priors['w1'][0] < w1 < priors['w1'][1] and 
            priors['w2'][0] < w2 < priors['w2'][1] and 
            priors['w3'][0] < w3 < priors['w3'][1]):
            return 0.0
    
    return -np.inf


def log_probability(params, t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr, priors, fit_planet_factor=False, planet_factor_fixed=0.):
    """
    Log probability function for MCMC (combines prior and likelihood).
    
    Parameters
    ----------
    params : list or array
        Model parameters
    t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr : array-like
        Model inputs and observed data
    priors : dict
        Dictionary with parameter bounds
    fit_planet_factor : bool, optional
        Whether planet_factor is a free parameter
        
    Returns
    -------
    float
        Log probability value
    """
    lp = log_prior(params, priors, fit_planet_factor)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr, fit_planet_factor, planet_factor_fixed)


def fit_lightcurve(t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr=None, priors=None, 
                  nwalkers=64, nsteps=2000, discard=100, fit_planet_factor=False, planet_factor_fixed=0.):
    """
    Fit light curve using MCMC.
    
    Parameters
    ----------
    t : array-like
        Time array
    lc_obs : array-like
        Observed light curve
    stellarmodel : array-like
        Stellar light curve model
    PC1, PC2, PC3 : array-like
        Principal components of systematic variations
    yerr : array-like, optional
        Measurement errors
    priors : dict, optional
        Parameter priors
    nwalkers : int, optional
        Number of MCMC walkers
    nsteps : int, optional
        Number of MCMC steps
    discard : int, optional
        Number of steps to discard as burn-in
    fit_planet_factor : bool, optional
        Whether to fit planet_factor as a free parameter (default: False)
    planet_factor_fixed : float, optional
        Fixed value for planet_factor when fit_planet_factor=False (default: 0)

    Returns
    -------
    tuple
        (samples, best_params) - MCMC samples and best-fit parameters
    """
    if yerr is None:
        yerr = np.ones_like(lc_obs) * 0.001  # Small default error
        
    if priors is None:
        # Default priors
        priors = {
            'A': (-0.1, 0.1),
            'B': (-0.1, 0.1),
            'P': (5, 15),
            'w0': (0.95, 1.05),
            'w1': (-10, 10),
            'w2': (-10, 10),
            'w3': (-10, 10),
        }
        if fit_planet_factor:
            priors['planet_factor'] = (0.0, 1.0)

    # Initial guess    
    if fit_planet_factor:
        initial = np.array([0.0, 0.0, 8.0, 1.0, 0, 0, 0, planet_factor_fixed])
        bounds = [priors['A'], priors['B'], priors['P'], priors['w0'], 
                  priors['w1'], priors['w2'], priors['w3'], priors['planet_factor']]
        ndim = 8
    else:
        initial = np.array([0.0, 0.0, 8.0, 1.0, 0, 0, 0])
        bounds = [priors['A'], priors['B'], priors['P'], priors['w0'], 
                  priors['w1'], priors['w2'], priors['w3']]
        ndim = 7

    # Least squares fitting
    print("Running least squares optimization...")
    
    result = minimize(chi2_func, initial, 
                     args=(t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr, fit_planet_factor, planet_factor_fixed),
                     method='powell', bounds=bounds)

    ls_params = result.x
    if not fit_planet_factor:
        ls_params = np.append(ls_params, planet_factor_fixed)
    
    print(f"Least squares chi2: {result.fun:.2f}")
    print(f"Least squares parameters: {ls_params}")

    # Setup walkers
    if fit_planet_factor:
        pos = ls_params + 1e-4 * np.random.randn(nwalkers, ndim)
    else:
        pos = ls_params[:ndim] + 1e-4 * np.random.randn(nwalkers, ndim)

    # Run MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(t, lc_obs, stellarmodel, PC1, PC2, PC3, yerr, priors, fit_planet_factor, planet_factor_fixed)
    )

    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Get samples and best fit
    samples = sampler.get_chain(discard=discard, flat=True)
    best_params = np.median(samples, axis=0)
    
    if not fit_planet_factor:
        best_params = np.append(best_params, planet_factor_fixed)

    return samples, best_params


def visualize_results(t, lc_obs, lc_err, best_params, stellarmodel, 
                     PC1, PC2, PC3, samples=None, output_path=None, fit_planet_factor=False):
    """
    Visualize fitting results.
    
    Parameters
    ----------
    t : array-like
        Time array
    lc_obs : array-like
        Observed light curve
    lc_err : array-like
        Measurement errors
    best_params : list or array
        Best-fit parameters
    stellarmodel : array-like
        Stellar light curve model
    PC1, PC2, PC3 : array-like
        Principal components of systematic variations
    samples : array-like, optional
        MCMC samples
    output_path : str, optional
        Path to save the figure
    fit_planet_factor : bool, optional
        Whether planet_factor was a free parameter
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Create figure with multiple panels
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Extract planet_factor from best_params
    if fit_planet_factor:
        planet_factor = best_params[7]
    else:
        planet_factor = best_params[7] if len(best_params) > 7 else 0.5

    # Best fit model
    ax1 = fig.add_subplot(gs[0:2, :])
    lc_best = lc_model(t, best_params, stellarmodel, PC1, PC2, PC3, fit_planet_factor, planet_factor)
    ax1.errorbar(t, lc_obs, yerr=lc_err, marker='.', color='k', alpha=0.3, label='Observed')
    ax1.plot(t, gf(lc_obs, 100), 'k-', alpha=0.3, label='Smoothed')
    ax1.plot(t, lc_best, 'r-', label='Best fit')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Light curve')
    ax1.legend()
    title_str = f'Light Curve Fit (planet_factor = {planet_factor:.3f})'
    ax1.set_title(title_str)

    # Components
    ax2 = fig.add_subplot(gs[2, :])
    if fit_planet_factor:
        A, B, P, w0, w1, w2, w3, planet_factor = best_params
    else:
        A, B, P, w0, w1, w2, w3 = best_params[:7]
        
    planetmodel = 1 + A * np.sin(2*np.pi*t/P) + B * np.cos(2*np.pi*t/P)
    systematic_model = w0 + w1*PC1 + w2*PC2 + w3*PC3

    astrophysical_model = (1-planet_factor) * stellarmodel + planet_factor * planetmodel
    ax2.plot(t, astrophysical_model, 'b-', label='Astrophysical model (planet + star)')
    ax2.plot(t, systematic_model, 'g-', label='Systematic model')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Model components')
    ax2.legend()
    ax2.set_title('Model Components')

    # Residuals
    ax3 = fig.add_subplot(gs[3, :])
    residuals = lc_obs - lc_best
    ax3.plot(t, residuals, 'k.', alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'Residuals (RMS = {np.std(residuals):.5f})')

    plt.tight_layout()
    plt.close("all")
    gc.collect()

    # Corner plot if samples provided
    if samples is not None:
        if fit_planet_factor:
            labels = ['A', 'B', 'P', 'w0', 'w1', 'w2', 'w3', 'planet_factor']
            truths = best_params
        else:
            labels = ['A', 'B', 'P', 'w0', 'w1', 'w2', 'w3']
            truths = best_params[:7]
            
        fig_corner = corner.corner(samples, labels=labels, truths=truths)
        
        if output_path:
            corner_path = output_path.replace('.png', '_corner.png')
            fig_corner.savefig(corner_path, dpi=150, bbox_inches='tight')
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def fourier_model(t, params):
    """
    First order Fourier series model.
    
    Parameters
    ----------
    t : array-like
        Time array
    params : list or array
        Model parameters [A, B, P, A2, B2, P2]
        
    Returns
    -------
    array-like
        Model values
    """
    A, B, P = params
    return A * np.sin(2*np.pi*t/P) + B * np.cos(2*np.pi*t/P) + 1


def fourier_chi2(params, t, lc):
    """
    Chi-squared function for Fourier model.
    
    Parameters
    ----------
    params : list or array
        Model parameters
    t : array-like
        Time array
    lc : array-like
        Light curve data
        
    Returns
    -------
    float
        Chi-squared value
    """
    model = fourier_model(t, params)
    return np.sum((lc - model)**2)


def fourier_log_prior(params, priors=None):
    """
    Log prior function for Fourier model.
    
    Parameters
    ----------
    params : list or array
        Model parameters [A, B, P]
    priors : dict, optional
        Dictionary with parameter bounds
        
    Returns
    -------
    float
        Log prior value (0 or -infinity)
    """
    A, B, P = params
    
    if priors is None:
        # Default priors
        if -1 < A < 1 and -1 < B < 1 and 0.1 < P < 100:
            return 0.0
    else:
        if (priors['A'][0] < A < priors['A'][1] and 
            priors['B'][0] < B < priors['B'][1] and 
            priors['P'][0] < P < priors['P'][1]):
            return 0.0
            
    return -np.inf


def fourier_log_probability(params, t, lc, yerr, priors=None):
    """
    Log probability function for Fourier model.
    
    Parameters
    ----------
    params : list or array
        Model parameters
    t : array-like
        Time array
    lc : array-like
        Light curve data
    yerr : array-like
        Measurement errors
    priors : dict, optional
        Dictionary with parameter bounds
        
    Returns
    -------
    float
        Log probability value
    """
    lp = fourier_log_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf

    model = fourier_model(t, params)
    chi2 = np.sum((lc - model)**2 / yerr**2)
    return lp - 0.5 * chi2


def fit_fourier(t, lc, yerr=None, priors=None, nwalkers=64, nsteps=2000, discard=500):
    """
    Fit time series to Fourier series.
    
    Parameters
    ----------
    t : array-like
        Time array
    lc : array-like
        Light curve data
    yerr : array-like, optional
        Measurement errors
    priors : dict, optional
        Dictionary with parameter bounds
    nwalkers : int, optional
        Number of MCMC walkers
    nsteps : int, optional
        Number of MCMC steps
    discard : int, optional
        Number of steps to discard as burn-in
        
    Returns
    -------
    tuple
        (samples, best_params, ls_params) - MCMC samples, best-fit parameters, and least squares parameters
    """
    # Initial guess
    initial = [0.1, 0.1, 8]

    if priors is None:
        # Default priors
        bounds = [(-1, 1), (-1, 1), (0.1, 20)]
    else:
        bounds = [priors['A'], priors['B'], priors['P']]

    # Least squares fit
    print("Running least squares...")
    result = minimize(fourier_chi2, initial, args=(t, lc), method='powell', bounds=bounds)
    ls_params = result.x
    print(f"Least squares parameters: A={ls_params[0]:.4f}, B={ls_params[1]:.4f}, P={ls_params[2]:.4f}")

    # MCMC
    if yerr is None:
        yerr = np.std(lc - fourier_model(t, ls_params)) * np.ones_like(lc)
        
    ndim = len(initial)
    pos = ls_params + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, fourier_log_probability, 
                                  args=(t, lc, yerr, priors))

    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=discard, flat=True)
    best_params = np.median(samples, axis=0)

    return samples, best_params, ls_params


def visualize_fourier_fit(t, lc, samples, best_params, ls_params, inject_params=None, output_path=None, n_samples=100):
    """
    Visualize Fourier fit results.
    
    Parameters
    ----------
    t : array-like
        Time array
    lc : array-like
        Light curve data
    samples : array-like
        MCMC samples
    best_params : list or array
        Best-fit parameters
    ls_params : list or array
        Least squares parameters
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    tuple
        (fig, fig_corner) - Figure objects
    """
    fig = plt.figure(figsize=(12, 10))

    # Best fit
    ax1 = plt.subplot(3, 1, 1)
    t_model = np.linspace(t.min(), t.max(), len(t)*5)
    model_best = fourier_model(t, best_params)
    model_best_plot = fourier_model(t_model, best_params)

    model_ls = fourier_model(t_model, ls_params)

    ax1.plot(t, lc, 'k.', alpha=0.2, label='Data')
    lc_bin = rebin(lc, 10)
    t_bin = rebin(t, 10)
    lc_bin_yerr = rebin_std(lc, 10) / np.sqrt(10)
    ax1.errorbar(t_bin, lc_bin, yerr=lc_bin_yerr, marker='o', ls='none', color='k', alpha=0.5, label='Binned data')

    # plot several samples from the MCMC chain
    sample_indices = np.random.choice(len(samples), n_samples, replace=False)
    for i in sample_indices:
        model_sample = fourier_model(t_model, samples[i])
        ax1.plot(t_model, model_sample, 'r-', alpha=0.3, lw=0.5)

    ax1.plot(t_model, model_best_plot, 'r-', label='MCMC best fit', lw=2)
    # ax1.plot(t_model, model_ls, 'b--', label='Least squares', lw=1.5)
    
    # plot the injected model if provided
    if inject_params is not None:
        model_inject = fourier_model(t_model, inject_params)
        ax1.plot(t_model, model_inject, 'b--', label='Injected model', lw=2)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flux')
    ax1.legend()
    ax1.set_title('Fourier Series Fit')

    # Residuals
    ax2 = plt.subplot(3, 1, 2)
    residuals = lc - model_best
    ax2.plot(t, residuals, 'k.', alpha=0.5)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Residuals (RMS = {np.std(residuals):.5f})')

    # Phase-folded
    ax3 = plt.subplot(3, 1, 3)
    phase = (t % best_params[2]) / best_params[2]

    phase_sorted_index = np.argsort(phase)
    phase_sorted = np.sort(phase)
    lc_phase_sorted = lc[phase_sorted_index]
    ax3.plot(phase_sorted, lc_phase_sorted, 'k.', alpha=0.3, label='Data')
    model_phase = fourier_model(phase_sorted * best_params[2], best_params)
    ax3.plot(phase_sorted, model_phase, 'r-', lw=2, label='Model')
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Flux')
    ax3.legend()
    ax3.set_title('Phase-folded Light Curve')

    plt.tight_layout()
    plt.close("all")
    gc.collect()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    # Corner plot
    labels = ['A', 'B', 'P']
    fig_corner = corner.corner(samples, labels=labels, truths=best_params)
    fig_corner.suptitle('Parameter Distributions', y=1.02)
    
    if output_path:
        corner_path = output_path.replace('.png', '_corner.png')
        fig_corner.savefig(corner_path, dpi=150, bbox_inches='tight')

    # Print results
    print("\nMCMC Results:")
    for i, label in enumerate(labels):
        q = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{label} = {q[1]:.4f} +{q[2]-q[1]:.4f} -{q[1]-q[0]:.4f}")

    return fig, fig_corner