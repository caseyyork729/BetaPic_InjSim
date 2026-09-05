from os import path, makedirs
import numpy as np
import matplotlib.pyplot as plt
import george
import emcee
import corner
from scipy.optimize import minimize
from george.kernels import Matern32Kernel
import h5py
from matplotlib.gridspec import GridSpec
from astropy.time import Time


# Function to create the intrinsic model (sine waves with known periods)
def intrinsic_model(t, params, periods):
    """
    Create intrinsic model for delta Scuti variability with known periods
    
    Parameters:
    ----------
    t : array
        Time array
    params : array
        Model parameters [A_1, B_1, A_2, B_2, ...] for each period
    periods : array
        Known periods for the delta Scuti star
    
    Returns:
    -------
    model : array
        Model flux values
    """
    model = np.zeros_like(t)
    for i, period in enumerate(periods):
        # For each period, add a sin + cos term
        # This is equivalent to A*sin(2πt/P + φ) but in the form you requested
        omega = 2 * np.pi / period
        A_i = params[2*i]
        B_i = params[2*i+1]
        model += A_i * np.sin(omega * t) + B_i * np.cos(omega * t)
    
    return model

# Function to create the combined model (intrinsic + GP)
class DeltaScutiSpaceTimeGPModel:
    def __init__(self, t, flux, flux_err, x_shift, y_shift, periods):
        """
        Initialize the delta Scuti GP model with space-time kernel
        
        Parameters:
        ----------
        t : array
            Time array
        flux : array
            Observed flux
        flux_err : array
            Flux errors
        x_shift : array
            X position shifts
        y_shift : array
            Y position shifts
        periods : array
            Known periods for the delta Scuti star
        """
        self.t = t
        self.flux = flux
        self.flux_err = flux_err
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.periods = periods
        
        # Number of sine wave parameters (2 for each period)
        self.n_wave_params = 2 * len(periods)
        
        # Normalize input dimensions to approximately the same scale
        # This helps with the optimization of length scales
        self.t_factor = 1.0 / np.std(t) if np.std(t) > 0 else 1.0
        self.x_factor = 1.0 / np.std(x_shift) if np.std(x_shift) > 0 else 1.0
        self.y_factor = 1.0 / np.std(y_shift) if np.std(y_shift) > 0 else 1.0
        
        # Initialize the kernel
        self.init_gp()
    
    def init_gp(self):
        """Initialize the Gaussian Process with space-time kernel"""
        # For the 3D Matern kernel, we'll use a different approach
        # Each dimension gets its own kernel, then we combine them
        
        # First, create independent kernels for each dimension
        # Use dimension-specific kernels with the 'axes' parameter
        time_kernel = Matern32Kernel(1.0, ndim=3, axes=0)  # operates on first dimension (time)
        
        x_kernel = Matern32Kernel(1.0, ndim=3, axes=1)     # operates on second dimension (x)
        y_kernel = Matern32Kernel(1.0, ndim=3, axes=2)     # operates on third dimension (y)
        
        # Combine them into a product kernel
        # This creates correlations across the full space-time domain
        self.kernel = time_kernel * x_kernel * y_kernel
        
        # Create the GP
        self.gp = george.GP(self.kernel)
    
    def set_params(self, params):
        """Set the model parameters"""
        # Extract sine wave parameters
        self.wave_params = params[:self.n_wave_params]
        
        # Extract kernel hyperparameters
        # Format: [kernel_amp, time_scale, x_scale, y_scale, white_noise]
        kernel_params = params[self.n_wave_params:]
        
        # Set kernel parameters
        # For the Matern32Kernel, we need to set the overall amplitude
        self.kernel.set_parameter_vector(kernel_params[:1])
        
        # Set the metric (different length scales for each dimension)
        # Note: In george, the metric is directly set on the kernel object
        # The metric is the inverse of the square of the length scales
        metric = np.array([
            1.0 / (kernel_params[1]**2),  # time_scale
            1.0 / (kernel_params[2]**2),  # x_scale
            1.0 / (kernel_params[3]**2)   # y_scale
        ])
        # In george, we set the metric directly as a parameter
        self.kernel.metric = metric
        
        # Set white noise (fifth kernel param)
        white_noise = kernel_params[4]
        
        # Create input array with normalized time and position coordinates
        X = np.vstack([
            self.t * self.t_factor,  
            self.x_shift * self.x_factor, 
            self.y_shift * self.y_factor
        ]).T
        
        # Set the GP's input coordinates and uncertainties
        try:
            self.gp.compute(X, np.sqrt(self.flux_err**2 + white_noise**2))
        except Exception as e:
            print(f"Error in compute: {e}")
            raise
    
    def get_input_coordinates(self):
        """Get the normalized input coordinates for the GP"""
        return np.vstack([
            self.t * self.t_factor,
            self.x_shift * self.x_factor,
            self.y_shift * self.y_factor
        ]).T
    
    def intrinsic_model(self):
        """Get the intrinsic model"""
        return intrinsic_model(self.t, self.wave_params, self.periods)
    
    def systematic_model(self):
        """Get the systematic model (GP prediction)"""
        # Compute residuals from intrinsic model
        residuals = self.flux - self.intrinsic_model()
        
        # Predict the systematic component using GP
        X = self.get_input_coordinates()
        mean, _ = self.gp.predict(residuals, X)
        return mean
    
    def full_model(self):
        """Get the full model (intrinsic + GP)"""
        return self.intrinsic_model() + self.systematic_model()
    
    def detrended_flux(self):
        """Get the detrended flux (original flux - systematic component)"""
        return self.flux - self.systematic_model()
    
    def log_likelihood(self, params):
        """Compute log likelihood for current parameters"""
        try:
            # Set the parameters
            self.set_params(params)
            
            # Compute residuals from intrinsic model
            intrinsic = self.intrinsic_model()
            residuals = self.flux - intrinsic
            
            # Compute log likelihood for the GP part (correlated noise)
            ll = self.gp.log_likelihood(residuals)
            
            # Add prior constraints (positivity of parameters)
            if not np.isfinite(ll):
                return -np.inf
            
            # Ensure all kernel parameters are positive
            if np.any(params[self.n_wave_params:] <= 0):
                return -np.inf
            
            return ll
        
        except Exception as e:
            print(f"Error in log_likelihood: {e}")
            return -np.inf
    
    def nll(self, params):
        """Negative log likelihood for optimization"""
        return -self.log_likelihood(params)
    
    def predict_systematic(self, t_new, x_new, y_new):
        """Predict systematic noise at new positions and times"""
        # First get the residuals from the current data
        residuals = self.flux - self.intrinsic_model()
        
        # Format the new prediction points
        X_new = np.vstack([
            t_new * self.t_factor,
            x_new * self.x_factor,
            y_new * self.y_factor
        ]).T
        
        # Then predict at new positions and times
        mean, var = self.gp.predict(residuals, X_new, return_var=True)
        return mean, np.sqrt(var)
    
    def predict_slices(self, n_points=50):
        """
        Generate predictions to visualize the model's behavior
        along different slices through the space-time domain
        
        Returns:
        -------
        t_grid : array
            Time grid for temporal slice
        x_grid : array
            X position grid for x slice
        y_grid : array
            Y position grid for y slice
        t_pred : array
            Predicted systematic noise along temporal slice
        x_pred : array
            Predicted systematic noise along x slice
        y_pred : array
            Predicted systematic noise along y slice
        """
        # Create a fine grid for time (holding x and y constant at their means)
        t_min, t_max = np.min(self.t), np.max(self.t)
        t_grid = np.linspace(t_min, t_max, n_points)
        x_const = np.mean(self.x_shift)
        y_const = np.mean(self.y_shift)
        
        # Create a fine grid for x position (holding time and y constant)
        x_min, x_max = np.min(self.x_shift), np.max(self.x_shift)
        x_grid = np.linspace(x_min, x_max, n_points)
        t_const = np.mean(self.t)
        
        # Create a fine grid for y position (holding time and x constant)
        y_min, y_max = np.min(self.y_shift), np.max(self.y_shift)
        y_grid = np.linspace(y_min, y_max, n_points)
        
        # Predict along temporal slice
        t_slice_mean, t_slice_std = self.predict_systematic(
            t_grid, 
            np.ones_like(t_grid) * x_const,
            np.ones_like(t_grid) * y_const
        )
        
        # Predict along x position slice
        x_slice_mean, x_slice_std = self.predict_systematic(
            np.ones_like(x_grid) * t_const,
            x_grid,
            np.ones_like(x_grid) * y_const
        )
        
        # Predict along y position slice
        y_slice_mean, y_slice_std = self.predict_systematic(
            np.ones_like(y_grid) * t_const,
            np.ones_like(y_grid) * x_const,
            y_grid
        )
        
        return t_grid, x_grid, y_grid, t_slice_mean, x_slice_mean, y_slice_mean, t_slice_std, x_slice_std, y_slice_std

# Function to run the MCMC optimization
def run_mcmc(model, initial_params, nwalkers=32, nsteps=2000, burn=500):
    """
    Run MCMC optimization
    
    Parameters:
    ----------
    model : DeltaScutiSpaceTimeGPModel
        Model instance
    initial_params : array
        Initial parameter values
    nwalkers : int
        Number of walkers
    nsteps : int
        Number of steps
    burn : int
        Number of burn-in steps
    
    Returns:
    -------
    samples : array
        MCMC samples
    """
    ndim = len(initial_params)
    
    # Initialize walkers with small perturbations around initial values
    pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)
    
    # Ensure all parameters are positive where needed
    for i in range(model.n_wave_params, ndim):
        pos[:, i] = np.abs(pos[:, i])
    
    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_likelihood)
    
    # Run burn-in
    print("Running burn-in...")
    pos, _, _ = sampler.run_mcmc(pos, burn, progress=True)
    sampler.reset()
    
    # Run production
    print("Running production...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    return sampler.get_chain(flat=True)

# Function to plot results
def plot_results(model, best_params, samples=None, outputDIR=None):
    """
    Plot the results of the GP fitting
    
    Parameters:
    ----------
    model : DeltaScutiSpaceTimeGPModel
        Model instance with optimized parameters
    best_params : array
        Best-fit parameters
    samples : array, optional
        MCMC samples for corner plot
    """
    # Set the optimized parameters
    model.set_params(best_params)
    # create a figure showing the detrended light curve and the intrinsic model
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(2, 1, 1)
    ax_res = fig.add_subplot(2, 1, 2, sharex=ax)  # Share x-axis with the first plot
    # Plot the detrended light curve
    ax.errorbar(model.t, model.detrended_flux(), yerr=model.flux_err, marker='.', ls='none', alpha=0.3, label='Detrended Flux')
    # Plot the intrinsic model
    ax.plot(model.t, model.intrinsic_model(), 'g-', lw=2, label='Intrinsic Model')    
    ax_res.errorbar(model.t, model.detrended_flux() - model.intrinsic_model(), 
                    yerr=model.flux_err, marker='.', ls='none', alpha=0.3)
    ax_res.set_xlabel('Time [hr]')
    ax.set_ylabel('Residual')
    ax.set_ylabel('Detrended Flux')
    ax.set_title('Detrended Light Curve and Intrinsic Model')
    if outputDIR is not None:
        # Save the figure if output directory is provided
        plt.savefig(f"{outputDIR}/detrended_light_curve.pdf")
        print(f"Saved detrended light curve plot to {outputDIR}/detrended_light_curve.pdf")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Original light curve and full model
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.errorbar(model.t, model.flux, yerr=model.flux_err, fmt='.k', alpha=0.3, label='Observed')
    ax1.plot(model.t, model.full_model(), 'r-', lw=2, label='Full Model')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flux')
    ax1.set_title('Original Light Curve and Full Model')
    ax1.legend()
    
    # Plot 2: Systematic noise component
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(model.t, model.systematic_model(), 'b-', lw=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Systematic Flux')
    ax2.set_title('Systematic Noise Component (GP Model)')
    
    # Visualize 1D slices through space-time domain
    t_grid, x_grid, y_grid, t_slice, x_slice, y_slice, t_std, x_std, y_std = model.predict_slices()
    
    # Plot position dependence
    ax3a = fig.add_subplot(3, 3, 7)
    ax3a.plot(x_grid, x_slice, 'g-', lw=2)
    ax3a.fill_between(x_grid, x_slice - t_std, x_slice + t_std, color='g', alpha=0.3)
    ax3a.set_xlabel('X Position')
    ax3a.set_ylabel('Systematic Flux')
    ax3a.set_title('X Position Dependence')
    
    ax3b = fig.add_subplot(3, 3, 8)
    ax3b.plot(y_grid, y_slice, 'g-', lw=2)
    ax3b.fill_between(y_grid, y_slice - y_std, y_slice + y_std, color='g', alpha=0.3)
    ax3b.set_xlabel('Y Position')
    ax3b.set_ylabel('Systematic Flux')
    ax3b.set_title('Y Position Dependence')
    
    # Plot time dependence
    ax3c = fig.add_subplot(3, 3, 9)
    ax3c.plot(t_grid, t_slice, 'g-', lw=2)
    ax3c.fill_between(t_grid, t_slice - t_std, t_slice + t_std, color='g', alpha=0.3)
    ax3c.set_xlabel('Time')
    ax3c.set_ylabel('Systematic Flux')
    ax3c.set_title('Time Dependence')
    
    plt.tight_layout()
    if outputDIR is not None:
        # Save the figure if output directory is provided
        plt.savefig(f"{outputDIR}/systematic_noise_components.pdf")
        print(f"Saved systematic noise components plot to {outputDIR}/systematic_noise_components.pdf")
    
    # Plot 2D visualization of systematic component
    fig = plt.figure(figsize=(15, 5))
    
    # Plot systematic vs. position colored by time
    ax1 = fig.add_subplot(1, 3, 1)
    sc = ax1.scatter(model.x_shift, model.y_shift, c=model.t, s=30, cmap='viridis')
    plt.colorbar(sc, ax=ax1, label='Time')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Observation Pattern (colored by time)')
    
    # Plot systematic vs. position
    ax2 = fig.add_subplot(1, 3, 2)
    sc = ax2.scatter(model.x_shift, model.y_shift, c=model.systematic_model(), s=30, cmap='coolwarm')
    plt.colorbar(sc, ax=ax2, label='Systematic Flux')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Systematic Noise vs. Position')
    
    # Plot systematic vs. time
    ax3 = fig.add_subplot(1, 3, 3)
    sc = ax3.scatter(model.t, model.systematic_model(), c=np.sqrt(model.x_shift**2 + model.y_shift**2), s=30, cmap='plasma')
    plt.colorbar(sc, ax=ax3, label='Radial Position')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Systematic Flux')
    ax3.set_title('Systematic Noise vs. Time')
    
    plt.tight_layout()
    if outputDIR is not None:
        # Save the figure if output directory is provided
        plt.savefig(f"{outputDIR}/systematic_vs_position_and_time.pdf")
        print(f"Saved systematic noise vs. position and time plot to {outputDIR}/systematic_vs_position_and_time.pdf")
    
    # Corner plot for MCMC samples if provided
    if samples is not None:
        # Create labels for the corner plot
        labels = []
        for i in range(len(model.periods)):
            labels.append(f"A_{i+1}")
            labels.append(f"B_{i+1}")
        
        # Add kernel parameter labels
        labels.extend(["GP_amp", "time_scale", "x_scale", "y_scale", "white_noise"])
        
        # If there are many parameters, select the most important ones
        if len(labels) > 10:
            # Choose a subset of parameters for clarity (e.g., first 2 periods and all kernel params)
            param_indices = list(range(4)) + list(range(len(labels)-5, len(labels)))
            selected_samples = samples[:, param_indices]
            selected_labels = [labels[i] for i in param_indices]
        else:
            selected_samples = samples
            selected_labels = labels
        
        # Thin the samples if there are too many
        max_samples = 5000
        if len(selected_samples) > max_samples:
            thin_factor = len(selected_samples) // max_samples
            selected_samples = selected_samples[::thin_factor]
        
        # Create the corner plot
        print("\nCreating corner plot of parameter posterior distributions...")
        fig = plt.figure(figsize=(12, 10))
        corner.corner(
            selected_samples, 
            labels=selected_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 8},
            title_fmt=".3f",
            fig=fig
        )
        if outputDIR is not None:
            # Save the corner plot if output directory is provided
            plt.savefig(f"{outputDIR}/corner_plot.pdf")
            print(f"Saved corner plot to {outputDIR}/corner_plot.pdf")
    plt.show()

# Main function to run the analysis
def main(t, flux, flux_err, x_shift, y_shift, periods, outputDIR=None,
         nwalkers=32, nsteps=2000, burn=500):
    """
    Main function to run the analysis
    
    Parameters:
    ----------
    t : array
        Time array
    flux : array
        Observed flux
    flux_err : array
        Flux errors
    x_shift : array
        X position shifts
    y_shift : array
        Y position shifts
    periods : array
        Known periods for the delta Scuti star
    nwalkers : int
        Number of walkers for MCMC
    nsteps : int
        Number of steps for MCMC
    burn : int
        Number of burn-in steps for MCMC
    
    Returns:
    -------
    model : DeltaScutiSpaceTimeGPModel
        Model instance with optimized parameters
    best_params : array
        Best-fit parameters
    samples : array
        MCMC samples
    """
    # Create the model
    model = DeltaScutiSpaceTimeGPModel(t, flux, flux_err, x_shift, y_shift, periods)
    
    # Initial guesses for sine wave parameters (amplitudes and phases)
    wave_params = np.zeros(2 * len(periods))
    for i in range(len(periods)):
        # Initial guess of amplitude = 0.1 * flux std
        wave_params[2*i] = 0.1 * np.std(flux)
        wave_params[2*i+1] = 0.1 * np.std(flux)
    
    # Initial guesses for kernel hyperparameters
    # [amplitude, time_scale, x_scale, y_scale, white_noise]
    kernel_params = np.array([
        np.var(flux),   # kernel amplitude
        1.0,            # time length scale
        1.0,            # x length scale
        1.0,            # y length scale
        np.median(flux_err)  # white noise
    ])
    
    # Combine parameters
    initial_params = np.concatenate([wave_params, kernel_params])
    
    # First, optimize using scipy
    print("Initial optimization...")
    result = minimize(model.nll, initial_params, method="L-BFGS-B", 
                      bounds=[(None, None)] * len(wave_params) + [(1e-10, None)] * len(kernel_params))
    
    opt_params = result.x
    print("Optimized parameters:", opt_params)
    
    # Run MCMC
    samples = run_mcmc(model, opt_params, nwalkers, nsteps, burn)
    
    # Get the best parameters (median of posterior)
    best_params = np.median(samples, axis=0)
    print("Best parameters from MCMC:", best_params)
    
    

    # save the modeling results into a h5py file if outputDIR is provided
    if outputDIR is not None:
        # Save the model parameters, samples, systematic model, intrinsic model, and detrended light curve to an HDF5 file
        outputFileName = f"{outputDIR}/model_results.h5"
        with h5py.File(outputFileName, 'w') as f:
            # Save the best parameters
            f.create_dataset('best_params', data=best_params)
            # Save the MCMC samples
            f.create_dataset('samples', data=samples)
            # Save the periods for reference
            f.create_dataset('periods', data=model.periods)
            # Save the intrinsic model values
            intrinsic_model_values = model.intrinsic_model()
            f.create_dataset('intrinsic_model', data=intrinsic_model_values)
            # Save the systematic model values
            systematic_model_values = model.systematic_model()
            f.create_dataset('systematic_model', data=systematic_model_values)
            # Save the detrended flux
            detrended_flux_values = model.detrended_flux()
            f.create_dataset('detrended_flux', data=detrended_flux_values)
            print(f"Saved modeling results to {outputFileName}")
    
    # Print the final parameters in a more readable format
    print("\nFinal Model Parameters:")
    print("-----------------------")
    
    # Print intrinsic model parameters
    for i, period in enumerate(periods):
        A_i = best_params[2*i]
        B_i = best_params[2*i+1]
        amplitude = np.sqrt(A_i**2 + B_i**2)
        phase = np.arctan2(B_i, A_i)
        print(f"Period {i+1} = {period:.6f}:")
        print(f"  A = {A_i:.6f}, B = {B_i:.6f}")
        print(f"  Amplitude = {amplitude:.6f}, Phase = {phase:.6f} rad")
    
    # Print kernel parameters
    kernel_start = len(periods) * 2
    kernel_params = best_params[kernel_start:]
    print("\nGP Space-Time Kernel Parameters:")
    print(f"  Amplitude = {kernel_params[0]:.6f}")
    print(f"  Time scale = {kernel_params[1]:.6f}")
    print(f"  X position scale = {kernel_params[2]:.6f}")
    print(f"  Y position scale = {kernel_params[3]:.6f}")
    print(f"  White noise = {kernel_params[4]:.6f}")
    
    # Calculate the proportion of variance explained by each component
    intrinsic_model_flux = model.intrinsic_model()
    systematic_model_flux = model.systematic_model()
    
    total_variance = np.var(model.flux)
    intrinsic_variance = np.var(intrinsic_model_flux)
    systematic_variance = np.var(systematic_model_flux)
    residual_variance = np.var(model.flux - model.full_model())
    
    print("\nVariance Analysis:")
    print(f"  Total flux variance: {total_variance:.6f}")
    print(f"  Intrinsic model variance: {intrinsic_variance:.6f} ({100*intrinsic_variance/total_variance:.1f}%)")
    print(f"  Systematic noise variance: {systematic_variance:.6f} ({100*systematic_variance/total_variance:.1f}%)")
    print(f"  Residual variance: {residual_variance:.6f} ({100*residual_variance/total_variance:.1f}%)")
    return model, best_params, samples

# Example usage
if __name__ == "__main__":

    saveFileName = '../Data/F410M/lightcurves/Host_Lightcurves_MP.h5'
    outputDIR = '../Data/F410M/lightcurves/stellar_model_results'
    if not path.exists(outputDIR):
        makedirs(outputDIR)
    Npt = 2200
    nsteps = 500
    nwalkers = 64
    with h5py.File(saveFileName, 'r') as hf:
        flux = hf['flux'][:]
        flux_err = hf['flux_err'][:]
        t_BJD = hf['t_BJD'][:]
        x_shift = hf['xshift'][:]
        y_shift = hf['yshift'][:]
        aperture_radii = hf['aperture_radii'][:]

    # Find indices for aperture = 25 and 30
    aperture_indices = []
    for radius in [30, 40]:
        idx = np.argmin(np.abs(aperture_radii - radius))
        aperture_indices.append(idx)

    # Convert BJD time to more readable format for plotting
    t_obj = Time(t_BJD, format='mjd')
    t_datetime = t_obj.datetime
    t_hours = (t_BJD - t_BJD[0]) * 24  # Hours from start
    periods = np.array([0.5049, 0.5284, 0.4835, 0.4424])

    # Normalize the flux
    norm_flux = flux[:, aperture_indices[0]] / np.median(flux[:, aperture_indices[0]])
    norm_flux_err = flux_err[:, aperture_indices[0]]/np.median(flux[:, aperture_indices[0]])
    
    # Run the analysis
    model, best_params, samples = main(t_hours[:Npt], norm_flux[:Npt], norm_flux_err[:Npt], x_shift[:Npt], y_shift[:Npt], periods, 
                                       outputDIR=outputDIR,
                                       nwalkers=32, nsteps=nsteps, burn=200)
    # Plot the results
    plot_results(model, best_params, samples, outputDIR=outputDIR)
    