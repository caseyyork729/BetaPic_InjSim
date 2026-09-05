#####################
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

stpsf_dir = os.path.expanduser("~/data/stpsf-data")

os.environ["STPSF_PATH"] = stpsf_dir
os.environ["WEBBPSF_PATH"] = stpsf_dir
os.environ["WEBBPSF_EXT_PATH"] = os.path.expanduser("~/jwst_refs/webbpsf_ext_data")
os.environ["PYSYN_CDBS"] = os.path.expanduser("~/jwst_refs/cdbs")
os.environ["CRDS_PATH"] = os.path.expanduser("~/jwst_refs/crds")
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
######################

import spaceKLIP
from scipy.ndimage import gaussian_filter
import glob
from astropy.io import fits
from os import path
import numpy as np
from pyklip import parallelized
#import os
#os.environ["OPENBLAS_NUM_THREADS"] = "1"

if __name__ == "__main__":
    filter_name = 'F410M'
    if filter_name == 'F210M':
        blur_FWHM = 2.432
    elif filter_name == 'F410M':
        blur_FWHM = 2.546901785046816
    input_dir = f'../Data/{filter_name}/psf_models'
    output_dir = f'../Data/{filter_name}/psf_models_blurred'
    hpf = True
    hpf_pix = 5
    if hpf:
        output_dir += f'_hp_npx_{hpf_pix}'
    if not path.exists(output_dir):
        os.mkdir(output_dir)
    

    
    gauss_sigma = blur_FWHM/ np.sqrt(8. * np.log(2.))
    print(hpf_pix, gauss_sigma)

    pid = 4758
    fnList = glob.glob(f'{input_dir}/*.fits')
    fnList.sort()
    for fn in fnList:
        print(f"Processing {fn}")
        with fits.open(fn, 'readonly') as f:
            model_psf = f[0].data
            fourier_sigma_size = (model_psf.shape[1] / hpf_pix) / (2. * np.sqrt(2. * np.log(2.)))
            if hpf:
                hpf_psf = parallelized.high_pass_filter_imgs(model_psf[None, :, :], numthreads=None, filtersize=fourier_sigma_size)
                blur_psf = gaussian_filter(hpf_psf[0, :, :], gauss_sigma)
            else:
                blur_psf = gaussian_filter(model_psf, gauss_sigma)
            f[0].data = blur_psf
            output_fn = path.join(output_dir, path.basename(fn))
            fits.writeto(output_fn, blur_psf, overwrite=True)