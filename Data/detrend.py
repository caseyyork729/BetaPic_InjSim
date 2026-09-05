import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

import george
from george import kernels

def polynomial_detrend(time, flux, x_shift, y_shift, degree=2, plot=True):
    """
    Remove correlations between flux and image shifts using polynomial detrending.
    
    Parameters:
    -----------
    time : array-like
        Time stamps of observations
    flux : array-like
        Normalized flux measurements
    x_shift : array-like
        X-shift of images in pixels
    y_shift : array-like
        Y-shift of images in pixels
    degree : int, optional
        Polynomial degree to use for the fit (default: 2)
    plot : bool, optional
        Whether to create diagnostic plots (default: True)
        
    Returns:
    --------
    corrected_flux : array-like
        Flux with shift correlations removed
    model : sklearn pipeline
        The fitted model for reference
    """
    # Create feature matrix (X and Y shifts)
    features = np.vstack([x_shift, y_shift]).T
    
    # Create and fit the polynomial model
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    model.fit(features, flux)
    
    # Predict the trend caused by shifts
    trend = model.predict(features)
    
    # Calculate the mean flux to maintain the same scale
    mean_flux = np.mean(flux)
    
    # Remove the trend, preserving the mean flux level
    corrected_flux = flux - trend + mean_flux
    
    if plot:
        plt.figure(figsize=(15, 12))
        
        # Original flux vs time
        plt.subplot(311)
        plt.scatter(time, flux, alpha=0.6, s=5, label='Original flux')
        plt.plot(time, trend, color='orange', label='Polynomial Trend', linewidth=2)
        plt.ylabel('Normalized Flux')
        plt.title('Original Flux Time Series')
        plt.legend()
        
        # Corrected flux vs time
        plt.subplot(312)
        plt.scatter(time, corrected_flux, alpha=0.6, s=5, color='g', label='Corrected flux')
        plt.ylabel('Normalized Flux')
        plt.title('Corrected Flux Time Series')
        plt.legend()
        
        # Original vs corrected flux
        plt.subplot(313)
        plt.scatter(flux, corrected_flux, alpha=0.6, s=5, color='r')
        plt.xlabel('Original Flux')
        plt.ylabel('Corrected Flux')
        plt.title('Original vs Corrected Flux')
        
        # Before and after correlation plots
        plt.figure(figsize=(15, 10))
        
        # Original correlations
        plt.subplot(221)
        plt.scatter(x_shift, flux, alpha=0.6, s=5)
        plt.xlabel('X Shift (pixels)')
        plt.ylabel('Original Flux')
        plt.title(f'Original Flux vs X-shift, r = {np.corrcoef(x_shift, flux)[0,1]:.3f}')
        
        plt.subplot(222)
        plt.scatter(y_shift, flux, alpha=0.6, s=5, color='r')
        plt.xlabel('Y Shift (pixels)')
        plt.ylabel('Original Flux')
        plt.title(f'Original Flux vs Y-shift, r = {np.corrcoef(y_shift, flux)[0,1]:.3f}')
        
        # Corrected correlations
        plt.subplot(223)
        plt.scatter(x_shift, corrected_flux, alpha=0.6, s=5)
        plt.xlabel('X Shift (pixels)')
        plt.ylabel('Corrected Flux')
        plt.title(f'Corrected Flux vs X-shift, r = {np.corrcoef(x_shift, corrected_flux)[0,1]:.3f}')
        
        plt.subplot(224)
        plt.scatter(y_shift, corrected_flux, alpha=0.6, s=5, color='r')
        plt.xlabel('Y Shift (pixels)')
        plt.ylabel('Corrected Flux')
        plt.title(f'Corrected Flux vs Y-shift, r = {np.corrcoef(y_shift, corrected_flux)[0,1]:.3f}')
        
        plt.tight_layout()
        plt.show()
        plt.close("all")
    
    return corrected_flux, model

# Example usage:
# corrected_flux, model = polynomial_detrend(time_array, flux_array, x_shift_array, y_shift_array, degree=3)




def gp_detrend(time, flux, x_shift, y_shift, plot=True):
    """
    Remove correlations between flux and image shifts using Gaussian Process regression.
    
    Parameters:
    -----------
    time : array-like
        Time stamps of observations
    flux : array-like
        Normalized flux measurements
    x_shift : array-like
        X-shift of images in pixels
    y_shift : array-like
        Y-shift of images in pixels
    plot : bool, optional
        Whether to create diagnostic plots (default: True)
        
    Returns:
    --------
    corrected_flux : array-like
        Flux with shift correlations removed
    gp : GaussianProcessRegressor
        The fitted Gaussian Process model
    """
    # Create feature matrix (X and Y shifts)
    features = np.vstack([x_shift, y_shift]).T
    
    # Standardize features for better GP performance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Define the GP kernel
    # We use a combination of:
    # - RBF kernel to model smooth variations
    # - WhiteKernel to account for noise in the measurements
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
    
    # Create and fit the GP model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp.fit(scaled_features, flux)
    
    # Predict the trend caused by shifts
    trend, _ = gp.predict(scaled_features, return_std=True)
    
    # Calculate the mean flux to maintain the same scale
    mean_flux = np.mean(flux)
    
    # Remove the trend, preserving the mean flux level
    corrected_flux = flux - trend + mean_flux
    
    if plot:
        plt.figure(figsize=(15, 12))
        
        # Original flux vs time
        plt.subplot(311)
        plt.scatter(time, flux, alpha=0.6, s=5, label='Original flux')
        plt.ylabel('Normalized Flux')
        plt.title('Original Flux Time Series')
        plt.legend()
        
        # Trend identified by GP
        plt.subplot(312)
        plt.scatter(time, trend, alpha=0.6, s=5, color='orange', label='GP Trend')
        plt.ylabel('Flux')
        plt.title('Identified Trend from Shifts')
        plt.legend()
        
        # Corrected flux vs time
        plt.subplot(313)
        plt.scatter(time, corrected_flux, alpha=0.6, s=5, color='g', label='Corrected flux')
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')
        plt.title('Corrected Flux Time Series')
        plt.legend()
        
        # Before and after correlation plots
        plt.figure(figsize=(15, 10))
        
        # Original correlations
        plt.subplot(221)
        plt.scatter(x_shift, flux, alpha=0.6, s=5)
        plt.xlabel('X Shift (pixels)')
        plt.ylabel('Original Flux')
        plt.title(f'Original Flux vs X-shift, r = {np.corrcoef(x_shift, flux)[0,1]:.3f}')
        
        plt.subplot(222)
        plt.scatter(y_shift, flux, alpha=0.6, s=5, color='r')
        plt.xlabel('Y Shift (pixels)')
        plt.ylabel('Original Flux')
        plt.title(f'Original Flux vs Y-shift, r = {np.corrcoef(y_shift, flux)[0,1]:.3f}')
        
        # Corrected correlations
        plt.subplot(223)
        plt.scatter(x_shift, corrected_flux, alpha=0.6, s=5)
        plt.xlabel('X Shift (pixels)')
        plt.ylabel('Corrected Flux')
        plt.title(f'Corrected Flux vs X-shift, r = {np.corrcoef(x_shift, corrected_flux)[0,1]:.3f}')
        
        plt.subplot(224)
        plt.scatter(y_shift, corrected_flux, alpha=0.6, s=5, color='r')
        plt.xlabel('Y Shift (pixels)')
        plt.ylabel('Corrected Flux')
        plt.title(f'Corrected Flux vs Y-shift, r = {np.corrcoef(y_shift, corrected_flux)[0,1]:.3f}')
        
        plt.tight_layout()
        plt.show()
        plt.close("all")
    
    return corrected_flux, gp

def gp_detrend_matern(time, flux, flux_err=None, x_shift=None, y_shift=None, nu=3/2, p0=None, 
                      optimization=True,
                      plot=True):
    """
    Remove correlations between flux and image shifts using Gaussian Process regression
    with Matérn kernel, implemented using the George package.
    
    Parameters:
    -----------
    time : array-like
        Time stamps of observations
    flux : array-like
        Normalized flux measurements
    flux_err : array-like, optional
        Flux measurement uncertainties
    x_shift : array-like
        X-shift of images in pixels
    y_shift : array-like
        Y-shift of images in pixels
    nu : float, optional
        Smoothness parameter for Matérn kernel (default: 3/2)
        Common values: 1/2 (exponential), 3/2, 5/2 (increasingly smooth)
    plot : bool, optional
        Whether to create diagnostic plots (default: True)
        
    Returns:
    --------
    corrected_flux : array-like
        Flux with shift correlations removed
    gp : george.GP
        The fitted Gaussian Process model
    """
    # Set default uncertainties if not provided
    if flux_err is None:
        flux_err = np.ones_like(flux) * np.std(flux) * 0.1
    
    # Create feature matrix (X and Y shifts)
    #features = np.vstack([time, x_shift, y_shift]).T
    features = np.vstack([x_shift, y_shift]).T
    
    # Standardize features for better GP performance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Define the kernel parameters
    # Initial metric - diagonal covariance matrix for anisotropic scaling
    metric = np.eye(2)  # 2D input (x_shift, y_shift)
    
    # Set Matérn kernel parameters based on nu
    if nu == 1/2:
        # Exponential kernel (Matérn 1/2)
        kernel = 0.1 * kernels.ExpKernel(metric=metric, ndim=2)
    elif nu == 3/2:
        # Matérn 3/2 kernel
        kernel = 0.1 * kernels.Matern32Kernel(metric=metric, ndim=2)
    elif nu == 5/2:
        # Matérn 5/2 kernel
        kernel = 0.1 * kernels.Matern52Kernel(metric=metric, ndim=2)
    
    # Add a white noise component
    
    
    # Create and initialize the GP model
    gp = george.GP(kernel, mean=np.mean(flux))
    gp.compute(scaled_features, flux_err)  # Compute the covariance matrix
    
    # Define the log likelihood function for optimization
    def neg_ln_like(p):
        # Update the kernel parameters
        gp.set_parameter_vector(p)
        # Return the negative log likelihood
        return -gp.log_likelihood(flux) 
    
    # Define the gradient of the log likelihood
    def grad_neg_ln_like(p):
        # Update the kernel parameters
        gp.set_parameter_vector(p)        
        # Return the gradient of the negative log likelihood
        return -gp.grad_log_likelihood(flux)
    
    # Optimize the kernel parameters
    print("Optimizing GP kernel parameters...")
    from scipy.optimize import minimize
    
    # Get the initial parameter vector
    initial_params = gp.get_parameter_vector()
    
    # Run the optimization
    if p0 is not None:
        initial_params = p0
    if optimization:
        result = minimize(neg_ln_like, initial_params, jac=grad_neg_ln_like, method="L-BFGS-B")
    
    # Update the model with the optimized parameters
    gp.compute(scaled_features, flux_err)
    gp.set_parameter_vector(result.x)
    
    # Print the optimized kernel parameters
    print("Optimized kernel parameters:")
    print(gp.get_parameter_dict())

    
    # Predict with the GP model (mean and variance)
    mean, var = gp.predict(flux, scaled_features, return_var=True)
    std = np.sqrt(var)
    
    # Calculate the mean flux to maintain the same scale
    flux_mean = np.mean(flux)
    
    # Remove the trend, preserving the mean flux level
    corrected_flux = flux - mean + flux_mean
    
    if plot:
        plt.figure(figsize=(15, 12))
        
        # Original flux vs time
        plt.subplot(311)
        plt.errorbar(time, flux, yerr=flux_err, fmt='.', alpha=0.6, color='orange', 
                    ecolor='orange', capsize=0, label='Original flux')
        plt.ylabel('Normalized Flux')
        plt.title('Original Flux Time Series')
        plt.legend()
        
        # Trend identified by GP
        plt.subplot(312)
        plt.errorbar(time, mean, fmt='.', alpha=0.6, color='blue', label='GP Trend')
        plt.fill_between(time, mean - 2*std, mean + 2*std, 
                         color='blue', alpha=0.2, label='±2σ uncertainty')
        plt.ylabel('Flux')
        plt.title(f'Identified Trend from Shifts (Matérn ν={nu})')
        plt.legend()
        
        # Corrected flux vs time
        plt.subplot(313)
        plt.errorbar(time, corrected_flux, yerr=flux_err, fmt='.', alpha=0.6, color='green', 
                    ecolor='green', capsize=0, label='Corrected flux')
        plt.ylabel('Normalized Flux')
        plt.xlabel('Time')
        plt.title('Corrected Flux Time Series')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        plt.close("all")
        
        # Before and after correlation plots
        plt.figure(figsize=(15, 10))
        
        # Original correlations
        plt.subplot(221)
        plt.scatter(x_shift, flux, alpha=0.6, s=5)
        plt.xlabel('X Shift (pixels)')
        plt.ylabel('Original Flux')
        plt.title(f'Original Flux vs X-shift, r = {np.corrcoef(x_shift, flux)[0,1]:.3f}')
        
        plt.subplot(222)
        plt.scatter(y_shift, flux, alpha=0.6, s=5, color='r')
        plt.xlabel('Y Shift (pixels)')
        plt.ylabel('Original Flux')
        plt.title(f'Original Flux vs Y-shift, r = {np.corrcoef(y_shift, flux)[0,1]:.3f}')
        
        # Corrected correlations
        plt.subplot(223)
        plt.scatter(x_shift, corrected_flux, alpha=0.6, s=5)
        plt.xlabel('X Shift (pixels)')
        plt.ylabel('Corrected Flux')
        plt.title(f'Corrected Flux vs X-shift, r = {np.corrcoef(x_shift, corrected_flux)[0,1]:.3f}')
        
        plt.subplot(224)
        plt.scatter(y_shift, corrected_flux, alpha=0.6, s=5, color='r')
        plt.xlabel('Y Shift (pixels)')
        plt.ylabel('Corrected Flux')
        plt.title(f'Corrected Flux vs Y-shift, r = {np.corrcoef(y_shift, corrected_flux)[0,1]:.3f}')
        
        plt.tight_layout()
        plt.show()
        plt.close("a")
    
    return corrected_flux, gp
# Example usage:
# corrected_flux, gp_model = gp_detrend(time_array, flux_array, x_shift_array, y_shift_array)
# compare_detrending_methods(time_array, flux_array, x_shift_array, y_shift_array)