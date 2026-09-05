import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from typing import Dict, Optional, Union, Tuple, Any
import gc

def perform_aperture_photometry(
    data: np.ndarray, 
    x_center: float, 
    y_center: float, 
    error: Optional[np.ndarray] = None,    
    aperture_radius: float = 5, 
    annulus_inner: float = 8,
    annulus_outer: float = 12,
    subtract_background: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform aperture photometry on an astronomical image with error estimation.
    
    Parameters
    ----------
    data : np.ndarray
        2D image data array for photometry
    x_center : float
        X-coordinate of the object center. If None, will attempt to find the brightest source.
    y_center : float
        Y-coordinate of the object center. If None, will attempt to find the brightest source.
    error : np.ndarray, optional
        2D error array corresponding to the data. If None, error will be estimated
        from the data using Poisson statistics and background noise.
    aperture_radius : float, default=5
        Radius of the photometry aperture in pixels.
    annulus_inner : float, default=8
        Inner radius of background annulus in pixels.
    annulus_outer : float, default=12
        Outer radius of background annulus in pixels.
    subtract_background : bool, default=True
        Whether to subtract background estimated from an annulus.
    verbose : bool, default=False
        Whether to print detailed results.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with photometry results including:
        - x_center, y_center: Source position
        - aperture_sum: Raw sum within aperture
        - background_sum: Estimated background contribution
        - net_flux: Background-subtracted flux
        - flux_error: Estimated error of the net flux
        - snr: Signal-to-noise ratio
        - aperture_radius, annulus_inner, annulus_outer: Parameters used
    
    Raises
    ------
    ValueError
        If no sources are detected when x_center and y_center are not provided.
    """
    # Calculate image statistics
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    if verbose:
        print(f"Image statistics: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")
    
    # If source coordinates are not provided, find the brightest source
    
    if verbose:
        print(f"Source position: x={x_center:.2f}, y={y_center:.2f}")
    
    # Define aperture and annulus for background
    aperture = CircularAperture((x_center, y_center), r=aperture_radius)
    annulus = CircularAnnulus((x_center, y_center), r_in=annulus_inner, r_out=annulus_outer)
    
    # Perform aperture photometry
    if error is not None:
        # Use provided error array
        phot_table = aperture_photometry(data, aperture, error=error)        
        has_error_array = True
    else:
        # No error array provided
        phot_table = aperture_photometry(data, aperture)        
        has_error_array = False
    
    # Calculate background
    if subtract_background:
        bkg_stats = ApertureStats(data, annulus)
        bkg_mean = bkg_stats.median  # Mean background in the annulus
        bkg_std = bkg_stats.std / np.sqrt(annulus.area) # Std of background in the annulus
        # Calculate the total background contribution within the aperture        
        bkg_sum = bkg_mean * aperture.area

        
        if verbose:
            print(f"Background mean: {bkg_mean:.4f} per pixel")
            print(f"Background std: {bkg_std:.4f} per pixel")
            print(f"Background subtracted: {bkg_sum:.4f} within aperture of area {aperture.area:.2f} pixels")
    else:
        # No background subtraction
        bkg_mean = 0.0
        bkg_std = 0.0
        bkg_sum = 0.0
    
    # Apply background subtraction to get final flux
    raw_flux = phot_table['aperture_sum'][0]
    final_flux = raw_flux - bkg_sum
    
    # Error estimation
    if has_error_array:
        # If error array was provided and used in aperture_photometry
        flux_error = phot_table['aperture_sum_err'][0]
        
        if subtract_background:
            # Add error from background subtraction (background noise × √N pixels)
            bkg_error = bkg_std * np.sqrt(aperture.area)
            # Combine errors in quadrature
            total_error = np.sqrt(flux_error**2 + bkg_error**2)
        else:
            total_error = flux_error
    else:
        # No error array provided, estimate from source counts and background
        # Poisson noise from source = √(source counts)
        source_error = np.sqrt(max(0, final_flux))  # Ensure positive value
        
        if subtract_background:
            # Background error (background noise × √N pixels)
            bkg_error = bkg_std * np.sqrt(aperture.area)
            # Combine errors in quadrature
            total_error = np.sqrt(source_error**2 + bkg_error**2)
        else:
            total_error = source_error
    
    # Calculate signal-to-noise ratio
    snr = final_flux / total_error if total_error > 0 else 0.0
    
    # Add results to the photometry table
    phot_table['background_mean'] = bkg_mean
    phot_table['background_std'] = bkg_std
    phot_table['background_sum'] = bkg_sum
    phot_table['net_flux'] = final_flux
    phot_table['flux_error'] = total_error
    phot_table['snr'] = snr
    
    # Display the results
    if verbose:
        print("\nPhotometry Results:")
        print("-" * 60)
        print(f"Aperture sum: {raw_flux:.2f}")
        if subtract_background:
            print(f"Background sum: {bkg_sum:.2f}")
        print(f"Net flux: {final_flux:.2f} ± {total_error:.2f}")
        print(f"Signal-to-noise ratio: {snr:.2f}")
        print("-" * 60)
    
    return {
        'x_center': x_center,
        'y_center': y_center,
        'aperture_sum': raw_flux,
        'background_mean': bkg_mean,
        'background_std': bkg_std,
        'background_sum': bkg_sum,
        'net_flux': final_flux,
        'flux_error': total_error,
        'snr': snr,
        'aperture_radius': aperture_radius,
        'annulus_inner': annulus_inner,
        'annulus_outer': annulus_outer
    }


def visualize_photometry(
    data: np.ndarray,
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Aperture Photometry'
) -> plt.Figure:
    """
    Visualize aperture photometry results.
    
    Parameters
    ----------
    data : np.ndarray
        2D image data array used for photometry
    results : Dict[str, Any]
        Dictionary with photometry results from perform_aperture_photometry
    figsize : Tuple[int, int], default=(10, 8)
        Figure size (width, height) in inches
    title : str, default='Aperture Photometry'
        Title for the plot
        
    Returns
    -------
    plt.Figure
        The created figure object
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Determine appropriate scaling
    vmin, vmax = np.percentile(data, [1, 99.5])
    
    # Plot image with apertures
    plt.imshow(data, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Create apertures for visualization
    aperture = CircularAperture(
        (results['x_center'], results['y_center']), 
        r=results['aperture_radius']
    )
    
    # Add annulus if background subtraction was done
    if 'annulus_inner' in results and 'annulus_outer' in results:
        annulus = CircularAnnulus(
            (results['x_center'], results['y_center']), 
            r_in=results['annulus_inner'], 
            r_out=results['annulus_outer']
        )
        annulus.plot(color='red', lw=2, label='Background Annulus')
    
    # Plot apertures
    aperture.plot(color='white', lw=2, label='Aperture')
    
    # Add colorbar and title
    plt.colorbar(label='Flux')
    plt.title(title)
    
    # Add result text
    info_text = (
        f"Net flux: {results['net_flux']:.2f} ± {results['flux_error']:.2f}\n"
        f"SNR: {results['snr']:.2f}"
    )
    
    # Add background info if available
    if results.get('background_sum', 0) > 0:
        info_text += f"\nBackground: {results['background_sum']:.2f}"
    
    plt.text(
        0.05, 0.95, 
        info_text,
        transform=plt.gca().transAxes, 
        color='white', 
        fontsize=12,
        bbox=dict(facecolor='black', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.legend(loc='lower right')
    return fig

def planet_aperture_photometry(
    data: np.ndarray,
    error: np.ndarray,
    x_center: float,
    y_center: float,
    aperture_radius: float = 5,
) -> Dict[str, Any]:
    """
    Wrapper function for performing aperture photometry on a planet-like source.
    
    This function uses the same parameters as perform_aperture_photometry but is 
    tailored for planet-like sources in exoplanet imaging.
    
    Parameters
    ----------
    data : np.ndarray
        2D image data array for photometry
    x_center : float
        X-coordinate of the planet center
    y_center : float
        Y-coordinate of the planet center
    aperture_radius : float, default=5
        Radius of the photometry aperture in pixels.
    
    Returns
    -------
    Dict[str, Any]
        Photometry results from perform_aperture_photometry
    """
    
    # Define aperture and annulus for background
    aperture = CircularAperture((x_center, y_center), r=aperture_radius, )
    # Use provided error array
    phot_table = aperture_photometry(data, aperture, error=error, method='exact')  
    flux = phot_table['aperture_sum'][0]
    err = phot_table['aperture_sum_err'][0] 
    return flux, err


# Example usage
if __name__ == "__main__":
    # Replace with your FITS file path
    image_file = "../Data/F210M/aligned/jw04758001001_03106_00001_nrca2_calints.fits"
    
    # Load data and error from file
    with fits.open(image_file) as hdul:
        data = hdul[1].data
        # If data is 3D, take the first slice
        if len(data.shape) == 3:
            data = data[1]
        
        # Try to get error array if available (often in extension 1)
        error = None
        if len(hdul) > 1 and hdul[2].data is not None:
            error = hdul[2].data
            if len(error.shape) == 3:
                error = error[1]
    
    radiusList = [15, 20, 25, 30, 35, 40]  # Example radii for aperture photometry
    fluxList = np.zeros(len(radiusList), dtype=float)
    fluxErrorList = np.zeros(len(radiusList), dtype=float)
    for i, radius in enumerate(radiusList):
        print(f"\nPerforming aperture photometry with radius: {radius}")    
        results = perform_aperture_photometry(
            data, 
            159.5,
            159.5,
            error=error, 
            aperture_radius=radius,
            annulus_inner=125,
            annulus_outer=135,
            subtract_background=True,
            verbose=True
        )
        fluxList[i] = results['net_flux']
        fluxErrorList[i] = results['flux_error']
    fig = plt.figure()
    plt.errorbar(radiusList, fluxList, yerr=fluxErrorList, fmt='.', label='Net Flux', color='blue', ls='none')
    ax2 = plt.gca().twinx()
    # Create a second y-axis for the signal to noise ratio
    snrList = fluxList / fluxErrorList
    # plot the signal to noise
    ax2.plot(radiusList, snrList, 'r-', label='SNR', alpha=0.5)
    ax2.set_ylabel('SNR', color='red')
    plt.xlabel('Aperture Radius (pixels)')
    plt.ylabel('Net Flux')
    plt.title('Aperture Photometry Results')


    # Visualize results
    fig = visualize_photometry(data, results)
    plt.show()
    plt.close("all")
    del fig, ax2
    gc.collect()
    