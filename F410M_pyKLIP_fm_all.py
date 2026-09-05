"""
Perform pyKLIP reduction on JWST coronagraphic images using RDI.
Includes forward modeling and MCMC fitting for planet position.

NIRCam V3 Angle: -0.55288201
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

############
stpsf_dir = os.path.expanduser("~/data/stpsf-data")

os.environ["STPSF_PATH"] = stpsf_dir
os.environ["WEBBPSF_PATH"] = stpsf_dir
os.environ["WEBBPSF_EXT_PATH"] = os.path.expanduser("~/jwst_refs/webbpsf_ext_data")
os.environ["PYSYN_CDBS"] = os.path.expanduser("~/jwst_refs/cdbs")
os.environ["CRDS_PATH"] = os.path.expanduser("~/jwst_refs/crds")
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
############

from os import path, makedirs
import pickle
import numpy as np
from astropy.io import fits
import argparse
from typing import Dict, Tuple, Any
import logging
import datetime

######################
# STPSF
import stpsf
from stpsf.constants import JWST_CIRCUMSCRIBED_DIAMETER

# Environment Variables
#from dataBase2pyKLIP import get_pyklip_filepaths
from spaceKLIP.pyklippipeline import get_pyklip_filepaths
#######################

from pyklip.instruments.JWST import JWSTData
from pyklip import parallelized, rdi
from pyklip.fmlib import fmpsf
import pyklip.fitpsf as fitpsf
import pyklip.fm as fm
import matplotlib.pyplot as plt

from createWebbPSF import generate_psf_model

np.seterr(divide='ignore', invalid='ignore')


# Filter configurations
FILTER_CONFIGS = {
    'F410M': {
        'pixscale': 0.06242419,  # from spaceKLIP database
        'PA_offset': 0,
        'psf_dir': f'../Data/F410M/psf_models_blurred_hp_npx_5',
        'psf_dict': {
            'jw04758001001_03106_00001_nrcalong_calints.fits': f'0',
            'jw04758001001_03107_00001_nrcalong_calints.fits': f'1',
            'jw04758001001_03108_00001_nrcalong_calints.fits': f'2',
            'jw04758001001_03109_00001_nrcalong_calints.fits': f'3',
            'jw04758001001_0310a_00001_nrcalong_calints.fits': f'4',
            'jw04758001001_0310b_00001_nrcalong_calints.fits': f'5',
            'jw04758001001_0310c_00001_nrcalong_calints.fits': f'6',
            'jw04758001001_0310d_00001_nrcalong_calints.fits': f'7',
            'jw04758001001_0310e_00001_nrcalong_calints.fits': f'8',
            'jw04758001001_0310f_00001_nrcalong_calints.fits': f'9',
            'jw04758002001_03106_00001_nrcalong_calints.fits': f'10',
            'jw04758002001_03107_00001_nrcalong_calints.fits': f'11',
            'jw04758002001_03108_00001_nrcalong_calints.fits': f'12',
            'jw04758002001_03109_00001_nrcalong_calints.fits': f'13',
            'jw04758002001_0310a_00001_nrcalong_calints.fits': f'14',
            'jw04758002001_0310b_00001_nrcalong_calints.fits': f'15',
            'jw04758002001_0310c_00001_nrcalong_calints.fits': f'16',
            'jw04758002001_0310d_00001_nrcalong_calints.fits': f'17',
            'jw04758002001_0310e_00001_nrcalong_calints.fits': f'18',
            'jw04758002001_0310f_00001_nrcalong_calints.fits': f'19',
            
            }
    },
}



# Planet configurations (based on best-fitting values)
""" 'b': {
            'sep': 544.055,    # Separation in mas
            'pa': 32.064,       # Position angle in degrees
            'raoff': 288.877,  # RA offset in mas
            'deoff':  461.068    # Dec offset in mas
        } """
PLANET_CONFIGS = {
    'b': {'sep': 544.055, 'pa': 31.09, 'guessflux': 420999.68},
}

def setup_logging(save_dir: str) -> None:
    """Setup logging configuration."""
    log_file = path.join(save_dir, 'reduction.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_wcs_cd(wcs: Any) -> np.ndarray:
    """Extract CD matrix from WCS header."""
    hd = wcs.to_header()
    cd = np.array([[hd['PC1_1'], hd['PC1_2']], 
                   [hd['PC2_1'], hd['PC2_2']]])
    return cd

def dataBase2JWSTData(database):
    """
    convert the databse to a JWSTData object so that we can use pyKLIP to process it
    """
    
    obskey = list(database.obs.keys())[0]        
    pt1, pt2 = get_pyklip_filepaths(database, obskey, return_maxbasis=False)    
    jwstDataObj = JWSTData(pt1, pt2, highpass=False)
    return jwstDataObj

def run_mcmc_fit(data_frame: np.ndarray, fm_frame: np.ndarray, 
                 data_cent: Tuple[float, float], fm_cent: Tuple[float, float],
                 guess_params: Dict[str, float], pixscale: float,
                 pa_offset: float = 0,
                 fittingSize=5) -> fitpsf.FMAstrometry:
    """Run MCMC fitting for planet position."""
    fit = fitpsf.FMAstrometry(guess_params['sep'], 
                             guess_params['pa'], 
                             fittingSize, 
                             method="mcmc")
    
    # Generate stamps
    fit.generate_fm_stamp(fm_frame, fm_cent, padding=5)
    fit.generate_data_stamp(data_frame, data_cent, dr=4, exclusion_radius=10)
    
    # Setup fitting parameters
    fit.set_kernel("matern32", [3.0], [r"$l$"])
    fit.set_bounds(1.0, 1.0, 2.0, [1.0])  # x, y, flux, corr_len ranges
    
    # Run MCMC
    fit.fit_astrometry(nwalkers=100, nburn=500, nsteps=1000, 
                       numthreads=1, save_chain=False)
    
    # Propagate errors
    fit.propogate_errs(star_center_err=0, 
                       platescale=pixscale, 
                       pa_offset=pa_offset, 
                       pa_uncertainty=0)
    
    return fit

def save_mcmc_plots(fit: fitpsf.FMAstrometry, plot_dir: str) -> None:
    """Save MCMC diagnostic plots."""
    # Chain plots
    fig = plt.figure(figsize=(10,8))
    chain = fit.sampler.chain
    
    params = [('RA', r"$\Delta$ RA"), 
              ('Dec', r"$\Delta$ Dec"), 
              ('Flux', r"$\alpha$"), 
              ('Corr', r"$l$")]
    
    for idx, (name, label) in enumerate(params, 1):
        ax = fig.add_subplot(411 + idx - 1)
        ax.plot(chain[:,:,idx-1].T, '-', color='k', alpha=0.3)
        ax.set_xlabel("Steps")
        ax.set_ylabel(label)
    
    fig.savefig(path.join(plot_dir, 'chain_plots.pdf'))
    plt.close(fig)
    
    # Corner plot
    fig = plt.figure()
    fig = fit.make_corner_plot(fig=fig)
    fig.savefig(path.join(plot_dir, 'corner_plots.pdf'))
    plt.close(fig)
    
    # Residuals plot
    fig = plt.figure()
    fig = fit.best_fit_and_residuals(fig=fig)
    fig.savefig(path.join(plot_dir, 'residual_images.pdf'))
    plt.close(fig)
    # save the 


def perform_klip_fm(database, 
                    filter_name,
                    planet_name,
                    mode='ADI + RDI',
                    numbasis=np.array([50]),
                    subsections=1,
                    output_dir=None):
    """
    Perform KLIP reduction on the dataset.
    Parameters
    ----------
    dataset : JWSTData
        The JWSTData object containing the data to be reduced.
    output_dir : str
        The directory where the reduced data will be saved.
    """
    # Set up the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Format mode name by replacing spaces with underscores
    mode_str = mode.replace(' ', '_')
    
    # Format numbasis for filename - join array elements with underscore
    numbasis_str = '_'.join(map(str, numbasis))
    
    # Create plot directory with mode and numbasis info
    plot_dir = path.join(output_dir, f'fm_plots_planet_{planet_name}_{mode_str}_klmodes_{numbasis_str}')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # Set up the PSF library
    filter_config = FILTER_CONFIGS[filter_name]
    planet_config = PLANET_CONFIGS[planet_name]
    webbPSF_dir = filter_config['psf_dir']
    # hard-code the psf number. The evolution of PSF during the observation should be small
    # Use the first one for all images
    webbPSF_num = 0
    webbPSF_FN = f'webbpsf_b_pic_b_F410M_{webbPSF_num}.fits'
    psf_data = fits.getdata(path.join(webbPSF_dir, webbPSF_FN))
    psfs = np.zeros((1, *psf_data.shape))
    psfs[0] = psf_data 

    # Calculate guess parameters
    pixelscale = filter_config['pixscale']
    pa_offset = filter_config['PA_offset']
    guess_params = {
        'sep': planet_config['sep'] / 1000 / pixelscale,
        'pa': planet_config['pa'],
        'flux': planet_config['guessflux']
    }
    dataset = dataBase2JWSTData(database)
    fm_class = fmpsf.FMPlanetPSF(dataset.input.shape,
                                 numbasis,  # numbasis
                                 guess_params['sep'],
                                 guess_params['pa'],
                                 guess_params['flux'],
                                 psfs,
                                 np.unique(dataset.wvs))
    
    
     # Run KLIP
    fileprefix = f'fm_planet_{planet_name}_{mode_str}_klmodes_{numbasis_str}'
    fm.klip_dataset(dataset,
                    fm_class,
                    outputdir=output_dir,
                    fileprefix=fileprefix,
                    annuli=[[0, guess_params['sep']-8],
                            [guess_params['sep']-8, guess_params['sep']+50],
                            [guess_params['sep']+50, 1000]],                            
                    subsections=subsections,
                    numbasis=numbasis,
                    psf_library=dataset.psflib,
                    aligned_center=dataset.psflib.aligned_center,
                    mode=mode,
                    highpass=False,
                    time_collapse='weighted-mean')
    # load the fm results
    # Load results
    fm_hdu = fits.open(path.join(output_dir, fileprefix + '-fmpsf-KLmodes-all.fits'))
    data_hdu = fits.open(path.join(output_dir, fileprefix + '-klipped-KLmodes-all.fits'))

    # run MCMC Fitting
    fit = run_mcmc_fit(
        data_hdu[0].data[0],
        fm_hdu[0].data[0],
        (data_hdu[0].header["PSFCENTX"], data_hdu[0].header["PSFCENTY"]),
        (fm_hdu[0].header['PSFCENTX'], fm_hdu[0].header['PSFCENTY']),
        guess_params,
        pixelscale,
        pa_offset=pa_offset,
    )
    guess_flux = planet_config['guessflux']
    fit_results = {
        'separation_mas': (fit.sep.bestfit * 1000, fit.sep.error * 1000),  # Convert to mas
        'pa_deg': (fit.PA.bestfit, fit.PA.error),
        'flux': (fit.fit_flux.bestfit * guess_flux, fit.fit_flux.error * guess_flux)
    }
    # Save fm PSFs for the next step
    # calcualte the PSF in each input image

    # Save MCMC plots
    save_mcmc_plots(fit, plot_dir)
    # save the fit_results into a txt file in the plot_dir
    # file name include the planet name, mode, numbasis, and today's date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    fit_results_file = path.join(plot_dir, f'fit_results_{planet_name}_{mode_str}_klmodes_{numbasis_str}_{date_str}.txt')
    with open(fit_results_file, 'w') as f:
        f.write("Parameter, Best Fit, Error\n")
        for key, value in fit_results.items():
            f.write(f"{key}={value[0]:.5f} +/- {value[1]:.5f}\n")

if __name__ == "__main__":
    import spaceKLIP
    data_root = '../Data/F410M_LIKELY_Th8'  # Base directory for data
    pid = 4758
    filter_name = 'F410M'
    subdir = 'coadded_hp_npx_5'
    mode = 'ADI + RDI'
    numbasis = 50
    database = spaceKLIP.database.create_database(
                                    input_dir=os.path.join(data_root, subdir),
                                    file_type='calints.fits',
                                    output_dir=data_root,                                    
                                    pid=pid)
    
    perform_klip_fm(database,
                    filter_name=filter_name,
                    planet_name='b',
                    mode=mode,
                    numbasis=np.array([numbasis]),
                    subsections=1,
                    output_dir=os.path.join(data_root, 'klipsub_fm_all_hp_npx_5'))