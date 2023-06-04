import meds
import psfex
import bfd
from bfd.keywords import *
from bfd import TierCollection
from bfd import Moment
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
from .read_meds_utils import Image, MOF_table, DetectionsTable,save_targets
from .utilities import save_obj, load_obj, bulkUnpack, rotate, angle, produce_sn, produce_cov, produce_mf
import glob
import numpy as np
import pandas as pd
import astropy.io.fits as fits
import ngmix.gmix as gmix
import ngmix           
import timeit
import pickle
import os
import argparse
import gc
import math
import time
import multiprocessing
from functools import partial
import sys      
import copy
from astropy import version 
import frogress
from .createTiers import createTiers








def make_targets(output_folder,**config):
    '''
    This stage of the pipeline loads into memory all the files with the measured targets moments and assemble them into one big file. It further assigns noise_tiers to each of the target.
    It works both on simulated and real data.
    '''

    # it begins checking if a number of keywords are provide din the config file. If not, it sets them up to the default values
    try:
        config['classes']
    except:
        print ('setting number of classes to default (7)')
        config['classes'] = 7
        
    try:
        config['re_run_noisetiers']
    except:
        print ('Re-running the noisetiers code')
        config['re_run_noisetiers'] = True
        
    try:
        config['noiseStep']
    except:
        print ('noiseStep not specified. I will set it up to 0.2 (default)')
        config['noiseStep'] = 0.2
    try:
        config['psfStep']
    except:
        print ('psfStep not specified. I will set it up to 0.1 (default)')
        config['psfStep'] = 0.1
    try:
        config['match_to_gold']
    except:
        print ('match_to_gold not specified. I will set it up to False (default)')
        config['match_to_gold'] = False
        
    doit=False
    
    # checks on MPI -----
    if config['MPI']:
        from mpi4py import MPI 
        comm = MPI.COMM_WORLD
        run_count =0
        
        if comm.rank==0:
            doit=True
    else:
        doit = True
        
        
    if doit:
        # the code needs to read in all the fits files, assign noisetiers, and save the final target file.
        # It distinguishes between image sims files and normal files.

        # let's copy a few parameters from the config file.
        params_template = {}
        params_template['n'] = config['n'] 
        params_template['sigma'] =  config['sigma']     

        # the code will loop over 3 types of files: data, sims +, sims -.
        add_labels = ['','ISp_','ISm_'] # if the label starts with IS, it refers to image-simulation files.
        
        # make the list for simulted tiles, they need to be loaded in the same order for noise cancelling.
        files_simulated_p = glob.glob(output_folder+'/targets/{1}{0}*.fits'.format('targets','ISp_'))
        files_simulated_m = [f.replace('ISp_','ISm_') for f in files_simulated_p]      
        
        #print (add_labels)
        for add in add_labels:
            nlost = 0

            for xx, tt in enumerate(['targets']): # this for loop is not very useful, but I'd like to keep it this way in case of future development of the code. 
                # Check if the FITS file for this combination of 'add' and 'tt' already exists
                if not os.path.exists(output_folder+'/{1}{0}_sample_g.fits'.format(tt, add)):

                    # Read all the target files based on the 'tt' and 'add' labels
                    files = glob.glob(output_folder+'/targets/{1}{0}*.fits'.format(tt, add))

                    # If 'add' label is 'ISp_', use files_simulated_p; if 'add' label is 'ISm_', use files_simulated_m
                    if add == 'ISp_':
                        files = files_simulated_p
                    if add == 'ISm_':
                        files = files_simulated_m

                    list_target = []

                    if len(files) > 0:
                        run_count = 0

                        hdulist = fits.HDUList([fits.PrimaryHDU()])
                        mute = fits.open(files[0])
                        names = [mute[1].columns[i].name for i in range(len(mute[1].columns))]

                        results_ = dict()

                        # Process the data from each file and store it in the 'results_' dictionary
                        for ii in frogress.bar(range(len(files))):
                            file = files[ii]
                            mute = fits.open(file)

                            for cname in mute[1].columns.names:
                                if ii == 0:
                                    results_[cname] = mute[1].data[cname]
                                else:
                                    try:
                                        if mute[1].data[cname].shape[1] > 1:
                                            results_[cname] = np.vstack([results_[cname], mute[1].data[cname]])
                                        else:
                                            results_[cname] = np.hstack([results_[cname], mute[1].data[cname]])
                                    except:
                                        results_[cname] = np.hstack([results_[cname], mute[1].data[cname]])

                        # Apply a mask to filter the data based on certain conditions
                        mask = (~np.isnan(results_['covariance'][:, 0])) & (results_['covariance'][:, 0] > 0.) & (results_['covariance'][:, 0] != np.inf)

                        if config['match_to_gold']:
                            gold = np.load('/global/cfs/cdirs/des/BFD_Y6/gold_id.npy', allow_pickle=True)
                            match = np.in1d(results_['id'], gold)
                            mask = mask & match

                        print(len(mask[mask]), len(mask))

                        # Create FITS columns for additional data, if required
                        cols = []
                        if not ('AREA' in names):
                            cols.append(fits.Column(name="AREA", format="K", array=0*np.ones_like(mask[mask])))
                        cols.append(fits.Column(name="NOISETIER", format="K", array=0*np.ones_like(mask[mask])))

                        new_cols = fits.ColDefs(cols)
                        hdu = fits.BinTableHDU.from_columns(mute[1].columns+new_cols)

                        # Copy necessary header keywords from the original file
                        for key in (hdrkeys['weightN'], hdrkeys['weightSigma']):
                            hdu.header[key] = mute[0].header[key]

                        # Copy processed data to the new FITS file
                        for cname in results_.keys():
                            hdu.data[cname] = results_[cname][mask]

                        # Copy necessary header keywords from the original file to the primary header
                        for key in (hdrkeys['weightN'], hdrkeys['weightSigma']):
                            hdulist[0].header[key] = mute[0].header[key]

                        hdulist[0].header['STAMPS'] = mute[0].header['STAMPS']

                        # Append the new HDU to the HDU list
                        hdulist.append(hdu.copy())
                        del hdu

                        try:
                            # Write the HDU list to the output FITS file
                            hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt, add))
                        except:
                            try:
                                # If the file write fails, try writing with the 'clobber' parameter set to True
                                hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt, add), clobber=True)
                            except:
                                pass    
        # MAKE NOISE TIERS *******************************************************************************
        
        # this understands if I need to run the noisetier snippet on the simulated or real data file.
        filex = (output_folder+'/{1}{0}_sample_g.fits'.format(tt,'ISp_'))
        if os.path.exists(filex):
            pass
        else:
            filex = (output_folder+'/{1}{0}_sample_g.fits'.format(tt,''))

        # Create the noisetier file if 're_run_noisetiers' flag is set and the file already exists, remove it
        if config['re_run_noisetiers']:
            if os.path.exists(output_folder+'/noisetiers.fits'):
                os.remove(output_folder+'/noisetiers.fits')
            tc = createTiers([filex], config['noiseStep'], config['psfStep'], config['Mf_min'], config['Mf_max'], config['sn_min'], config['sn_max'])
            tc.save(output_folder+'/noisetiers.fits')
        else:
            # Create the noisetier file if it doesn't already exist
            if not os.path.exists(output_folder+'/noisetiers.fits'):
                tc = createTiers([filex], config['noiseStep'], config['psfStep'], config['Mf_min'], config['Mf_max'], config['sn_min'], config['sn_max'])
                tc.save(output_folder+'/noisetiers.fits')


        # ASSIGN NOISE TIERS ********************
        from bfd import TierCollection

        tc = TierCollection.load(output_folder+'/noisetiers.fits')
        
        # here need to load it and assign noise tiers to the fits file again ---
        for add in add_labels:
            nlost = 0
            hdulist = fits.HDUList([fits.PrimaryHDU()])
                    
            if os.path.exists(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add)):
                print ('saving')
                for xx,tt in enumerate(['targets']):
                    m_ = fits.open(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add))
                    for key in (hdrkeys['weightN'],hdrkeys['weightSigma']):
                        hdulist[0].header[key] = m_[0].header[key]
                    hdulist[0].header['STAMPS'] = m_[0].header['STAMPS']
            
            
                    n_ = tc.assign(m_[1].data['covariance'])
                    noise_tiers = np.zeros(m_[1].data['covariance'].shape[0]).astype(np.int)
                    for key in n_.keys():
                        noise_tiers[n_[key]]=key
                        
                    

                    # dismiss stuff that doesn't pass the cuts ***********
                    
                    mask = (produce_sn(m_)>config['sn_min']) & (produce_sn(m_)< config['sn_max']) & (produce_mf(m_)>config['Mf_min']) & (produce_mf(m_)< config['Mf_max']) 

                    # add class column for later **********************************
                    # we want to use the worst noise tiers for this!!

                    phi = -0.5*(angle(m_[1].data['moments'].T))
                    e_,o_ = rotate(m_[1].data['moments'].T,m_[1].data['moments'][:,:2].T,phi)
                    new_DV_targets = np.hstack([e_,o_])

                    #cholesky decomposition of the best tier
                    nt = fits.open(output_folder+'/noisetiers.fits')
                    u,o = bulkUnpack(nt[-1].data['COVARIANCE'][:,:])
                    u[-1],o[-1]
                    new_cov = np.zeros((7,7))
                    new_cov[:5,:5] = u[1]
                    new_cov[5:,5:] = o[1]                  

                    L = np.linalg.cholesky(np.linalg.inv(new_cov))

                    # project the DV
                    new_DV_targets = np.matmul(new_DV_targets,L)
                    #
                    mask__ = np.array([True,True,True,False,False,False,False])
                    new_DV_targets = new_DV_targets[:,mask__]

                    sn = produce_sn(m_)

                    # a upper cut on sn must be based on the worst tier. a lowe cut on sn must be based on worst tier.
                    u,o = bulkUnpack(nt[1].data['COVARIANCE'][:,:])
                    mask_l = m_[1].data['moments'][:,0]/np.sqrt(u[1,0,0]) > 2 

                    # upper cut **
                    u,o = bulkUnpack(nt[-1].data['COVARIANCE'][:,:])
                    mask_u = m_[1].data['moments'][:,0]/np.sqrt(u[1,0,0]) < 70

                    mask_total = mask_u & mask_l

                    #class_ = (new_DV_targets/config['classes']).astype(np.int)
                    
                    bins_ = np.exp(np.linspace(0,7,config['classes']))
                    bins_[0] = -100
                    class_ = np.digitize(new_DV_targets,bins_)

                   
                    
                    
                    class_u = np.array(['{0}_{1}_{2}'.format(x[0],x[1],x[2]) for x in class_])
                    mask_total = mask_u & mask_l #  
                    class_u[~mask_total] = '-100_-100_-100'
               
                    
                
                    # regroup classes - can't have a class with less than 10 % of the targets.
                    
                    try:
                        cols_ = fits.Column(name="class",format="128A",array=class_u)
                        hdu = fits.BinTableHDU.from_columns(m_[1].columns+cols_)
                    except:
                        hdu = fits.BinTableHDU.from_columns(m_[1].columns)
                        hdu.data['class'] = class_u
                    hdu.data['NOISETIER'] = noise_tiers
                    
                    print('NOISETIER')
                    print(hdu.data['NOISETIER'])
                    print('--------------')
                    
                    # ++++++++++++++
                    m__ = (~mask) & (hdu.data['AREA']==0)
                    hdu.data['AREA'][m__] = 1

                    for key in (hdrkeys['weightN'],hdrkeys['weightSigma']):
                        hdu.header[key] = m_[0].header[key]
                        
                    hdulist.append(hdu.copy())

                        
                        
                        
                        

                    # Next create an image of mean covariance matrix for each tier
                    hdr = hdulist[0].header
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
                                data[i,j] = np.mean(m_[1].data['covariance'][mask,index])
                                data[j,i] = np.mean(m_[1].data['covariance'][mask,index])
                                index += 1
                        hdu = fits.ImageHDU(data)
                        hdu.header[hdrkeys['weightN']] =  m_[1].header[hdrkeys['weightN']]
                        hdu.header[hdrkeys['weightSigma']] =m_[1].header[hdrkeys['weightSigma']]
                        hdu.header['TIERNAME'] = tier
                        hdu.header['TIERLOST'] = 0
                        # Record mean covariance of odd moments in header

                        
                        '''
                        # saved in order M0xM0, M0xMR, M0xM1, M0xM2, M0xMC,
                        # MRxMR, MRxM1, MRxM2, MRxMC, M1xM1, M1xM2, M1xMC,
                        # M2xM2, M2xMC, MCxMC (total of 15)
                        
                        '''
                        xx =0.5*np.mean(m_[1].data['covariance'][:,1]+m_[1].data['covariance'][:,2])
                        #0.5*(m_[1].data['covariance'][:,3])
                        yy = 0.5*np.mean(m_[1].data['covariance'][:,1]-m_[1].data['covariance'][:,2])
                        
                        #xyCov = 0.5 * np.array( [[ cov[m.M0,m.MR]+cov[m.M0,m.M1], cov[m.M0,m.M2] ],
                        #     [ cov[m.M0, m.M2], cov[m.M0,m.MR]-cov[m.M0,m.M1]]] )
                            
                            
                        
                        mean_c = 0.5*(yy+xx)
                        hdu.header['COVMXMX'] = mean_c
                        hdu.header['COVMXMY'] = 0.5*np.mean(m_[1].data['covariance'][:,3])
                        hdu.header['COVMYMY'] = mean_c
                        hdulist.append(hdu.copy())
                        

                        del hdu
                    # assign noisetiers -1
     
                    mask = noise_tiers ==-1
                    if len(mask[mask])>0:
                        cvm = [hdulist[ii+2].data[0,0] for ii in range(len(hdulist)-2)]
                        ttu = np.array([hdulist[ii+2].header['TIERNAME'] for ii in range(len(hdulist)-2)])
                        hdulist[1].data['NOISETIER'][mask] =  ttu[np.array([np.argmin((u-cvm)**2) for u in produce_cov(hdulist)[mask]])]
        

                        
                    try:
                        hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add))
                    except:
                        try:
                            hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add),overwrite = True)# 
                        except:
                            pass
          
        
    if config['MPI']:
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
