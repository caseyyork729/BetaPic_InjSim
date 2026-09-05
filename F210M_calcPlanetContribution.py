"""
This script runs after pyKLIP_fm_all.py, pyKLIP_fm_all.py script provides the PA, sep, and the scaling of the planet PSF

The purpose of this script is to calculate pixel-by-pixel conribution of the planet.

Results from this script will be used to select pixels for extracting the planetary light curves.

"""

import os

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from spaceKLIP.psf import JWST_PSF

from scipy.ndimage import fourier_shift, shift

# Results from pyKLIP_fm_all.py run: May 7, 2025
# separation_mas=541.96546 +/- 0.66366
# pa_deg=31.54918 +/- 0.07349
# flux=1455230.39320 +/- 25034.89895

""" separation_mas=546.13130 +/- 0.64051
pa_deg=33.48438 +/- 0.06700
flux=1495830.20665 +/- 24575.38739 """

FILTER_CONFIGS = {
    'F210M': {
        'pixscale': 0.03068,
        'PA_offset': 0,
        'separation': 546.13130,
        'separation_err': 0.64051,
        'pa': 33.48438,
        'pa_err': 0.06700,
        'flux': 1495830.20665,
        'psf_dir': f'../Data/F210M/psf_models_blurred',
        'psf_dict': {
            'jw04758001001_03106_00001_nrca2_calints.fits': f'0',
            'jw04758001001_03107_00001_nrca2_calints.fits': f'1',
            'jw04758001001_03108_00001_nrca2_calints.fits': f'2',
            'jw04758001001_03109_00001_nrca2_calints.fits': f'3',
            'jw04758001001_0310a_00001_nrca2_calints.fits': f'4',
            'jw04758001001_0310b_00001_nrca2_calints.fits': f'5',
            'jw04758001001_0310c_00001_nrca2_calints.fits': f'6',
            'jw04758001001_0310d_00001_nrca2_calints.fits': f'7',
            'jw04758001001_0310e_00001_nrca2_calints.fits': f'8',
            'jw04758001001_0310f_00001_nrca2_calints.fits': f'9',
            'jw04758002001_03106_00001_nrca2_calints.fits': f'10',
            'jw04758002001_03107_00001_nrca2_calints.fits': f'11',
            'jw04758002001_03108_00001_nrca2_calints.fits': f'12',
            'jw04758002001_03109_00001_nrca2_calints.fits': f'13',
            'jw04758002001_0310a_00001_nrca2_calints.fits': f'14',
            'jw04758002001_0310b_00001_nrca2_calints.fits': f'15',
            'jw04758002001_0310c_00001_nrca2_calints.fits': f'16',
            'jw04758002001_0310d_00001_nrca2_calints.fits': f'17',
            'jw04758002001_0310e_00001_nrca2_calints.fits': f'18',
            'jw04758002001_0310f_00001_nrca2_calints.fits': f'19',
            
            }
    },
}



def calcPlanetContribution(sci_file, psf_file, filter_name, savedir):
    """
    Expand and shift the PSF to the specified pixel coordinates.
    
    Parameters
    ----------
    """
    hdu = fits.open(sci_file)
    data = hdu[1].data[0]
    phead = hdu[0].header
    shead = hdu[1].header
    # get the roll angle used in pyKLIP 
    # These are kept consistent with pyKLIP set-ups
    PA = shead['ROLL_REF'] - (shead['V3I_YANG']*shead['VPARITY'])
    pixscale = np.sqrt(shead['PIXAR_A2'])
    ny, nx = data.shape
    apername = phead['AperName']
    date = phead['DATE-BEG']
    hdu.close()

    psf_hdu = fits.open(psf_file)
    psf = psf_hdu[0].data
    psf_hdu.close()

    ny_psf, nx_psf = psf.shape
    
    # expand the PSF to the size of the science image
    psf_expanded = np.zeros((ny, nx))
    psf_expanded[:ny_psf, :nx_psf] = psf

    # calculate the shift
    planet_config = FILTER_CONFIGS[filter_name]
    PA_image = planet_config['pa'] - PA
    separation_px = planet_config['separation'] / pixscale / 1000
    dx_sep = -separation_px * np.sin(np.deg2rad(PA_image))
    dy_sep = separation_px * np.cos(np.deg2rad(PA_image))
    xc = nx / 2 - 0.5
    yc = ny / 2 - 0.5
    xc_psf = nx_psf / 2 - 0.5
    yc_psf = ny_psf / 2 - 0.5
    xc_planet = xc + dx_sep
    yc_planet = yc + dy_sep
    print("Planet position in the image: ", PA, separation_px, xc_planet, yc_planet)
    x_shift = xc_planet - xc_psf
    y_shift = yc_planet - yc_psf
    # shift the PSF
    # print the amount of shift needed
    # print(f'Shift amount: {x_shift:.2f}, {y_shift:.2f}')
    yxshift = np.array([y_shift, x_shift])
    psf_shift = planet_config['flux'] * shift(psf_expanded, yxshift, order=3)
    # normalize the PSF


    planetContribution = psf_shift / data
    # save the planet contribution
    savefn = sci_file.replace('.fits', f'_planet_contribution.fits')
    savefn = os.path.join(savedir, os.path.basename(savefn))
    fits.writeto(savefn, planetContribution, overwrite=True)
    psf_model_fn = sci_file.replace('.fits', f'_psf_model.fits')
    psf_model_fn = os.path.join(savedir, os.path.basename(psf_model_fn))
    fits.writeto(psf_model_fn, psf_shift, overwrite=True)
    # visualize the location of the planet
    fig, ax = plt.subplots()
    vmin = np.nanpercentile(data, 0.5)
    vmax = np.nanpercentile(data, 99.95)
    ax.imshow(data, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    ax.plot(xc_planet, yc_planet, 'ro', markersize=5,
            mfc='none', mew=0.5)
    ax.plot(xc, yc, 'bx', markersize=5)
    ax.set_title(f'Roll angle: {PA:.2f} deg\n')
    # save the plot in the same directory
    plotfn = sci_file.replace('.fits', f'_planet_contribution.png')
    plotfn = os.path.join(savedir, os.path.basename(plotfn))
    plt.savefig(plotfn, dpi=300)
    return planetContribution

if __name__ == "__main__":
    filter_name = 'F210M'
    import spaceKLIP
    data_root = '../Data/F210M_LIKELY_Th8'  # Base directory for data
    pid = 4758
    filter_name = 'F210M'
    subdir = 'coadded'    
    savedir = os.path.join(data_root, 'planet_contribution')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    database = spaceKLIP.database.create_database(
                                    input_dir=os.path.join(data_root, subdir),
                                    file_type='calints.fits',
                                    output_dir=data_root,                                    
                                    pid=pid)
    key = list(database.obs.keys())[0]
    type = database.obs[key]['TYPE'].value
    sci_files = database.obs[key]['FITSFILE'].value[type == 'SCI']
    planet_config = FILTER_CONFIGS[filter_name]
    for sci_file_i in sci_files:
        print(f'Processing {sci_file_i}')
        webbPSF_num = planet_config['psf_dict'][os.path.basename(sci_file_i)]
        webbPSF_FN = f'webbpsf_b_pic_b_F210M_{webbPSF_num}.fits'                                            
        psf_file_i = os.path.join(planet_config['psf_dir'], webbPSF_FN)
        calcPlanetContribution(sci_file_i, psf_file_i, filter_name, savedir)