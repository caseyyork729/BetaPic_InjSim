"""
Generate WebbPSF models for HR8799b for multiple JWST observations.
Creates PSF models accounting for the coronagraphic mask offsets in each observation.
Supports multiple filters with different pixel scales.
"""

import os
import numpy as np
from astropy.io import fits

#####################
stpsf_dir = os.path.expanduser("~/data/stpsf-data")

os.environ["STPSF_PATH"] = stpsf_dir
os.environ["WEBBPSF_PATH"] = stpsf_dir
os.environ["WEBBPSF_EXT_PATH"] = os.path.expanduser("~/jwst_refs/webbpsf_ext_data")
os.environ["PYSYN_CDBS"] = os.path.expanduser("~/jwst_refs/cdbs")
os.environ["CRDS_PATH"] = os.path.expanduser("~/jwst_refs/crds")
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
######################

import spaceKLIP
from spaceKLIP import utils as ut
from spaceKLIP.psf import JWST_PSF

# Filter configurations
FILTER_CONFIGS = {
    'F210M': {
        'pixscale': 0.03068,  # arcsec/pixel
        'aperture': 'NRCA2_MASK335R',
        'fov_pix': 65
    },
    'F410M': {
        'pixscale': 0.06242419,  # arcsec/pixel
        'aperture': 'NRCA5_MASK335R',
        'fov_pix': 65
    }
}

def setup_planet_parameters(filter_name):
    """
    Define the Beta Pic planet parameters and convert to pixel coordinates.
    
    Parameters
    ----------
    filter_name : str
        Name of the filter being used (e.g., 'F200W', 'F444W')
    
    Returns
    -------
    dict
        Dictionary containing pixel-scale converted parameters for each planet
    """
    # beta Pic b
    planet_dict = {
        'b': {
            'sep': 544.055,    # Separation in mas
            'pa': 32.064,       # Position angle in degrees
            'raoff': 288.877,  # RA offset in mas
            'deoff':  461.068    # Dec offset in mas
        }}
    
    # Get pixel scale for specified filter
    try:
        pxsc_arcsec = FILTER_CONFIGS[filter_name]['pixscale']
    except KeyError:
        raise ValueError(f"Filter {filter_name} not supported. Available filters: {list(FILTER_CONFIGS.keys())}")
    
    # Convert to pixel coordinates
    for planet in planet_dict:
        raoff_pix = planet_dict[planet]['raoff'] / 1000 / pxsc_arcsec
        deoff_pix = planet_dict[planet]['deoff'] / 1000 / pxsc_arcsec
        
        planet_dict[planet].update({
            'dx_pix': raoff_pix,
            'dy_pix': deoff_pix,
            'sep_pix': np.sqrt(raoff_pix**2 + deoff_pix**2),
            'pa_deg': np.rad2deg(np.arctan2(raoff_pix, deoff_pix)),
            'pxsc': pxsc_arcsec
        })
    
    return planet_dict

def calculate_mask_offsets(sci_file, roll_ref, planet_params):
    """
    Calculate the offset between planet and coronagraphic mask.
    
    Parameters
    ----------
    sci_file : str
        Path to science FITS file
    roll_ref : float
        Roll angle reference in degrees
    planet_params : dict
        Planet parameters including pixel coordinates
    
    Returns
    -------
    tuple
        Separation and position angle relative to mask
    """
    _, _, _, _, _, _, _, maskoffs = ut.read_obs(sci_file)
    
    if maskoffs is None:
        return None, None
    
    # Mask offsets (negative because of coordinate convention)
    mask_xoff = -maskoffs[:, 0]
    mask_yoff = -maskoffs[:, 1]
    
    # Rotate by roll angle and flip RA axis
    roll_rad = np.deg2rad(roll_ref)
    mask_raoff = -(mask_xoff * np.cos(roll_rad) - mask_yoff * np.sin(roll_rad))
    mask_deoff = mask_xoff * np.sin(roll_rad) + mask_yoff * np.cos(roll_rad)
    
    # Calculate planet position relative to mask
    sim_dx = planet_params['dx_pix'] - mask_raoff
    sim_dy = planet_params['dy_pix'] - mask_deoff
    
    # Convert to arcsec and degrees
    sim_sep = np.median(np.sqrt(sim_dx**2 + sim_dy**2) * planet_params['pxsc'])
    sim_pa = np.median(np.rad2deg(np.arctan2(sim_dx, sim_dy)))
    
    return sim_sep, sim_pa

def generate_psf_model(sci_file, roll_ref, sim_sep, sim_pa, filter_name, oversample=2):
    """
    Generate WebbPSF model for given parameters.
    
    Parameters
    ----------
    sci_file : str
        Science FITS file path
    roll_ref : float
        Roll angle reference in degrees
    sim_sep : float
        Separation in arcsec
    sim_pa : float
        Position angle in degrees
    filter_name : str
        Name of the filter being used
    
    Returns
    -------
    numpy.ndarray
        PSF model
    """
    filter_config = FILTER_CONFIGS[filter_name]
    apername = filter_config['aperture']
    date = fits.getheader(sci_file, 0)['DATE-BEG']
    
    offsetpsf_func = JWST_PSF(
        apername,
        filter_name,
        date=date,
        fov_pix=filter_config['fov_pix'],
        oversample=oversample,
        sp=None,
        use_coeff=False
    )
    
    return offsetpsf_func.gen_psf(
        [sim_sep, sim_pa],
        mode='rth',
        PA_V3=roll_ref,
        do_shift=False,
        quick=False,
        addV3Yidl=False
    )

def main(filter_name='F410M', data_root=None, planet='b'):
    """
    Main function to generate PSF models for all observations.
    
    Parameters
    ----------
    filter_name : str
        Filter to use for PSF generation ('F200W' or 'F444W')
    data_root : str, optional
        Root directory for data. If None, uses default path
    """
    if data_root is None:
        data_root = f'../Data/{filter_name}'
    
    pid = 4758
    database = spaceKLIP.database.create_database(
        input_dir=os.path.join(data_root, 'aligned'),
        file_type='calints.fits',
        output_dir=data_root,
        pid=pid
    )
    
    # Get planet parameters for specified filter
    planet_dict = setup_planet_parameters(filter_name)
    
    # Create output directory
    output_dir = os.path.join(data_root, f'psf_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each observation
    for key in database.obs.keys():
        # Find science exposures
        ww_sci = np.where(database.obs[key]['TYPE'] == 'SCI')[0]
        print(f"Found {len(ww_sci)} science exposures for {key}")
        for ww in ww_sci:
            print(f"Processing exposure {ww}")
            sci_file = database.obs[key]['FITSFILE'][ww]
            roll_ref = database.obs[key]['ROLL_REF'][ww]
        
            
        
            # Calculate offsets
            sim_sep, sim_pa = calculate_mask_offsets(sci_file, roll_ref, planet_dict[planet])
            if sim_sep is None:
                print(f"Skipping {key}: No mask offsets found")
                continue
            print(f"Simulated separation: {sim_sep:.2f} arcsec, PA: {sim_pa:.2f} degrees")
            
            # Generate PSF model
            psf_model = generate_psf_model(sci_file, roll_ref, sim_sep, sim_pa, filter_name)
        
            # Save PSF model
            output_name = f"webbpsf_b_pic_{planet}_{filter_name}_{ww}.fits"
            output_path = os.path.join(output_dir, output_name)
        
            fits.writeto(output_path, psf_model, overwrite=True)
            print(f"Saved PSF model: {output_name}")

if __name__ == "__main__":
    # Example usage for different filters
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate WebbPSF models for HR8799 planets')
    parser.add_argument('--filter', choices=['F210M', 'F410M'], default='F410M',
                      help='Filter to use for PSF generation')
    parser.add_argument('--data-root', type=str, help='Root directory for data')
    parser.add_argument('--planet', type=str, help='which planet to use', default='b')
    
    args = parser.parse_args()
    
    main(filter_name=args.filter, data_root=args.data_root, planet=args.planet)