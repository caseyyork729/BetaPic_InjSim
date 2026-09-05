"""
Perform pyKLIP reduction on one chunk of images

Limit the function for using RDI for now
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pickle
import numpy as np

from pyklip import parallelized, rdi

def get_wcs_cd(wcs):
    hd = wcs.to_header()
    cd = np.array([[hd['PC1_1'], hd['PC1_2']], [hd['PC2_1'], hd['PC2_2']]])
    return cd

if __name__ == "__main__":

    chunNum = 200
    NINTS = 10
    filt = 'F210M'
    use_simplePSFs = True
    pickleFileDIR = f'../Data/{filt}/timeseries_prep_NINTS_{NINTS}/'
    if filt == 'F210M':
        fileKey = f'JWST_NIRCAM_NRCA2_{filt}_MASKRND_MASK335R_SUB320A335R_NINT_{NINTS}_'
    else:
        fileKey = f'JWST_NIRCAM_NRCALONG_{filt}_MASKRND_MASK335R_SUB320A335R_NINT_{NINTS}_'
    
    pickleFileName = pickleFileDIR + f'{fileKey}{chunNum:03d}.pkl'
    # Load the pickled data
    with open(pickleFileName, 'rb') as f:
        data = pickle.load(f)

    data.input = data._input
    data.centers = data._centers
    data.filenames = data._filenames
    data.filenums = data._filenums
    data.PAs = data._PAs
    data.wvs = data._wvs
    data.wcs = data._wcs
    cd = get_wcs_cd(data.wcs[0])
    for i in range(len(data.wcs)):
        data.wcs[i].wcs.cd = cd
    data.IWA = data._IWA
    data.OWA = data._OWA    
    if use_simplePSFs:
        psflibDataFileName = pickleFileDIR + f'{fileKey}simplepsflib.pkl'
    else:
        psflibDataFileName = pickleFileDIR + f'{fileKey}psflib.pkl'

    saveDIR = f'../Data/{filt}/klipsub/chunk_{NINTS}_{chunNum:03d}/'
    if not os.path.exists(saveDIR):
        os.makedirs(saveDIR)

    preparedPSFLib = os.path.join(saveDIR, f'{fileKey}psflib.pkl')

    # Check if the PSF library has already been prepared
    if os.path.exists(preparedPSFLib):
        print(f'PSF library already prepared for {pickleFileName}')
        with open(preparedPSFLib, 'rb') as f:
            psflib = pickle.load(f)
        psflib.prepare_library(data)
    else:
        print(f'Preparing PSF library for {pickleFileName}')
        # prepare the PSF Library
        # Load the PSFlib
        with open(psflibDataFileName, 'rb') as f:
            psflib_data = pickle.load(f)

        psflib_data_all = np.append(psflib_data.master_library, data._input, axis=0)
        psflib_filenames_all = np.append(psflib_data.master_filenames, data._filenames, axis=0)
        aligned_center = psflib_data.aligned_center        
        psflib = rdi.PSFLibrary(psflib_data_all, aligned_center, psflib_filenames_all, compute_correlation=True, highpass=psflib_data.highpass)
        psflib.prepare_library(data)
        print("save the prepared PSF library")
        # save the prepared PSF library
        with open(preparedPSFLib, 'wb') as f:
            pickle.dump(psflib, f)

    # save prepared PSF library
    
    parallelized.klip_dataset(data,
                              mode='RDI',
                              annuli=3,
                              subsections=1,
                              numbasis=[1, 5, 10, 15, 20],
                              psf_library=psflib,
                              outputdir=saveDIR,
                              )

