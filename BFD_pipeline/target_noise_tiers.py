from astropy.table import QTable, Table, vstack
import astropy.io.fits as fits
import numpy as np
import glob
import os
import frogress
import bfd
from bfd import TierCollection
from bfd.keywords import *
from .healpy_routines import *

# Add option to save flux covariance maps.

def target_noise_tiers( **config):
    """
    This function reads target files and creates two new FITS files: 
    one for Bayesian integration and another to store useful quantities for the catalog.
    """
    if 'gold_id' not in config.keys():
        config['gold_id'] = None
        print ('You did not provide any list of gold catalog IDs; I will set the variable gold_id to None, which means non of the targets will be dropped from the catalog.')
        
    if 'star_galaxy_separation_value' not in config.keys():
        config['star_galaxy_separation_value'] = 1000
        print ('You did not provide any value for the variable star_galaxy_separation_value; I will set the variable star_galaxy_separation_value to 1000 (which in practice means none of the targets will be flagged as stars')
    
    if 'overwrite_noisetiers' not in config.keys():
        config['overwrite_noisetiers'] = False
        print ('You did not provide any value for the variable overwrite_noisetiers; I will set the variable to False')

    if 'flux_covariance_maps_nside' not in config.keys():
        config['flux_covariance_maps_nside'] = False
        print ('You did not provide any value for the variable flux_covariance_maps_nside; I will set the variable to False, which means I will not produce healpy maps of the flux covariances')
  
    if 'sn_min' not in config.keys():
        config['sn_min'] = 7
        print ('You did not provide any value for the variable sn_min; I will set the variable to 7')
  
    if 'sn_max' not in config.keys():
        config['sn_max'] = 200
        print ('You did not provide any value for the variable sn_max; I will set the variable to 200')
  
    if 'psfStep' not in config.keys():
        config['psfStep'] = 0.3
        print ('You did not provide any value for the variable psfStep; I will set the variable to 0.3')
    
    if 'noiseStep' not in config.keys():
        config['noiseStep'] = 0.3
        print ('You did not provide any value for the variable noiseStep; I will set the variable to 0.3')
        
        
  
    # Globbing to find all target files in the specified output folder
    f = glob.glob(config['output_folder'] + '/targets/*')

    # Defining columns to keep for integration and for the final catalog
    columns_to_keep_for_integration = ['id', 'moments', 'covariance']
    columns_to_keep_for_final_catalog = ['id', 'ra', 'dec', 'psf_moments',
                                       'psf_hsm_moments', 'Mf_per_band', 'cov_Mf_per_band', 
                                       'mfrac_per_band', 'bkg']

    # Initializing tables for integration and final catalog
    table_for_integration = QTable()
    table_for_final_catalog = QTable()

    print ('reading files')
    for i in frogress.bar(range(len(f))):
        f_ = f[i]
        hdul = fits.open(f_)
        try:
            data = QTable(hdul[1].data)

            # Selecting data for integration and final catalog
            selected_for_integration = data[columns_to_keep_for_integration]
            selected_for_final_catalog = data[columns_to_keep_for_final_catalog]

            # Background correction for the Flux Covariance
            correction = (1. + 1. / np.sqrt(data['pixels_used_for_bkg']))**2
            data['covariance'][:, 0] *= correction

            # Stacking the selected data for integration and final catalog
            table_for_integration = vstack([table_for_integration, selected_for_integration])  
            table_for_final_catalog = vstack([table_for_final_catalog, selected_for_final_catalog])  

            # save a full covariance map. -----------------------------------------------
            
            
            # save a flux covariances map. ----------------------------------------------
            if config['flux_covariance_maps_nside']:
                number_of_fluxes = table_for_final_catalog['Mf_per_band'].shape[1]
                if i == 0:
                    counts_map  = np.zeros(12*config['flux_covariance_maps_nside']**2)
                    flux_covariance_map = np.zeros((12*config['flux_covariance_maps_nside']**2,number_of_fluxes))

                pix = convert_to_pix_coord(table_for_final_catalog['ra'], table_for_final_catalog['dec'], nside = config['flux_covariance_maps_nside'])
                unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
                counts_map[unique_pix] += np.bincount(idx_rep, weights=np.ones(len(pix)))
                for flux_index in range(number_of_fluxes):
                    flux_covariance_map[unique_pix,flux_index] += np.bincount(idx_rep, weights=table_for_final_catalog['cov_Mf_per_band'][:,flux_index])
        except:
            print ('failed file ',f_) 
        # save a flux covariances map.
    if config['flux_covariance_maps_nside']:
        for flux_index in range(number_of_fluxes):
            flux_covariance_map[counts_map != 0,flux_index] /= counts_map[counts_map!=0]
        np.save(config['output_folder'] + '/flux_covariances_map',flux_covariance_map)
            

    # Matching to gold catalog if provided
    if config['gold_id'] is not None:
        gold_id = np.load(config['gold_id'], allow_pickle=True)
        mask_gold = np.in1d(table_for_final_catalog['id'], gold_id)
        table_for_integration = table_for_integration[mask_gold]
        table_for_final_catalog = table_for_final_catalog[mask_gold]

    # Applying star-galaxy separation to reduce catalog size before integration
    if config['star_galaxy_separation_value'] is not None:
        mask_sg = (table_for_integration['moments'][:, 1] / table_for_integration['moments'][:, 0] < config['star_galaxy_separation_value'])
        table_for_integration = table_for_integration[mask_sg]
        table_for_final_catalog = table_for_final_catalog[mask_sg]                          

    

    # Building noise tiers if necessary
    compute_noise_tiers = (os.path.exists(config['output_folder'] + '/noisetiers.fits') and config['overwrite_noisetiers']) or not os.path.exists(config['output_folder'] + '/noisetiers.fits')
    if compute_noise_tiers:
        # Randomly sampling covariance for noise tier construction
        size = hdulist1[1].data['covariance'].shape[0]
        keep = np.random.random(size=size) < 0.01
        covs = hdulist1[1].data['covariance'][keep]

        # Building tier set from the collected covariances
        print("# Building tiers from total of", covs.shape[0], "targets")
        print(config['sn_min'], config['sn_max'])

        tc = bfd.TierCollection(covs, wtN=4, wtSigma=config['filter_sigma'],
                                snMin=config['sn_min'], snMax=config['sn_max'],
                                fluxMin=None, fluxMax=None,
                                stepA=[config['noiseStep'], config['psfStep']],
                                minTargets=10)

        # Saving the constructed tiers
        tc.save(config['output_folder']+'/noisetiers.fits')

        
        
        
        
    
    # assign noisetiers to targets.       
    tc = TierCollection.load(config['output_folder'] +'/noisetiers.fits')
    n_ = tc.assign(table_for_integration['covariance'])
    noise_tiers = -1*np.ones(table_for_integration['covariance'].shape[0]).astype(np.int)
    for key in n_.keys():
        noise_tiers[n_[key]]=key
        
    table_for_integration.add_column(noise_tiers, name='NOISETIER')
    # Creating FITS HDU objects for both tables
    hdu1 = fits.BinTableHDU(table_for_integration)
    hdu2 = fits.BinTableHDU(table_for_final_catalog)

    # Adding primary HDU and headers to the HDU lists
    hdulist1 = fits.HDUList([fits.PrimaryHDU(), hdu1])
    hdulist2 = fits.HDUList([fits.PrimaryHDU(), hdu2])
    hdulist1[1].header[hdrkeys['weightN']] = 4
    hdulist1[1].header[hdrkeys['weightSigma']] = config['filter_sigma']
    hdulist2[1].header[hdrkeys['weightN']] = 4
    hdulist2[1].header[hdrkeys['weightSigma']] = config['filter_sigma']

    
    # add entries for each noisetier in the fits file.
    unique_tiers = np.unique(noise_tiers)
    unique_tiers = unique_tiers[unique_tiers != -1]

    for tier in unique_tiers:
        # Build covariance matrix for even parity
        mask = noise_tiers == tier
        nn = 5
        data = np.zeros((nn,nn))
        index = 0

        for i in range(nn):
            for j in range(i,nn):
                data[i,j] = np.mean(hdulist1[1].data['covariance'][mask,index])
                data[j,i] = np.mean(hdulist1[1].data['covariance'][mask,index])
                index += 1

        hdu = fits.ImageHDU(data)
        hdu.header[hdrkeys['weightN']] =  hdulist1[1].header[hdrkeys['weightN']]
        hdu.header[hdrkeys['weightSigma']] = hdulist1[1].header[hdrkeys['weightSigma']]
        hdu.header['TIERNAME'] = tier
        hdu.header['TIERLOST'] = 0
        # Record mean covariance of odd moments in header

        xx =0.5*np.mean(hdulist1[1].data['covariance'][:,1]+hdulist1[1].data['covariance'][:,2])
        yy = 0.5*np.mean(hdulist1[1].data['covariance'][:,1]-hdulist1[1].data['covariance'][:,2])                        
        mean_c = 0.5*(yy+xx)
        hdu.header['COVMXMX'] = mean_c
        hdu.header['COVMXMY'] = 0.5*np.mean(hdulist1[1].data['covariance'][:,3])
        hdu.header['COVMYMY'] = mean_c
        hdulist1.append(hdu)

    # save
    hdulist1.writeto(config['output_folder'] +'/targets.fits', overwrite=True)
    hdulist2.writeto(config['output_folder'] +'/targets_extrainfo.fits', overwrite=True)
    
    