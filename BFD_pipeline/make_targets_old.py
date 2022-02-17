import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
from .read_meds_utils import Image, MOF_table, DetectionsTable,save_targets
from .utilities import update_progress, save_obj, load_obj
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmix.gmix as gmix
import ngmix           
import timeit
import pickle
import os
import argparse
import gc
import math
import time, sys
import multiprocessing
from functools import partial
import sys      
import copy
from astropy import version 
import astropy.io.fits as fits
import frogress


'''
We can write v2 - in the cpp file, we make 1 file per noisetier? or should we make it here?
It look to me that making the change gere was better

'''
def make_targets(output_folder,**config):
    if config['MPI']:
        from mpi4py import MPI 
    comm = MPI.COMM_WORLD
    if comm.rank==0:
        
        # the code needs to read in all the fits files, assign noisetiers, and save the final target file.
        # It distinguishes between image sims files and normal files.
        params_template = {}
        params_template['n'] = config['n'] 
        params_template['sigma'] =  config['sigma']  

        # make the sigma_Mf bins.
        bins_Mf = 10**(np.linspace(np.log10(config['sigmaM_min']**2+0.00001),np.log10(config['sigmaM_max']**2),config['steps']+1))
        bins_Mf[0] = config['sigmaM_min']**2
   
        add_labels = ['','ISp_','ISm_']
        
        # make the list for simulted tiles, they need to be loaded in the same order for noise cancelling.
        
        files_simulated_p = glob.glob(output_folder+'/targets/{1}{0}*.fits'.format('targets','ISp_'))
        files_simulated_m = [f.replace('ISp_','ISm_') for f in files_simulated_p]      
        
        
        files_simulated_p
        for add in add_labels:
            nlost = 0
            for xx,tt in enumerate(['targets']):
                # read all the target files 
                files = glob.glob(output_folder+'/targets/{1}{0}*.fits'.format(tt,add))
                
                if add =='ISp_':
                    files = files_simulated_p
                if add =='ISm_':
                    files = files_simulated_m
                list_target = []



                if len(files)>0:
                    run_count = 0



                    hdulist = pf.HDUList([pf.PrimaryHDU()])
                    # assign noisetiers **********************************************************************
                    file_removed = False
                  
                    for ii in frogress.bar(range(len(files))):
                        file = files[ii]
                        try:
                            mute = pf.open(file)
                        except:
                            os.remove(file)
                            file_removed = True
                        if ii ==0:
                            noisetier = np.digitize(mute[1].data['cov_even'][:,0], bins_Mf)
                            occ  = mute[1].data['cov_even'][:,0]
                            mf  = mute[1].data['moments'][:,0]
                        else:
                            noisetier = np.hstack([noisetier,np.digitize(mute[1].data['cov_even'][:,0], bins_Mf)])
                            occ = np.hstack([occ,mute[1].data['cov_even'][:,0]])
                            mf = np.hstack([mf,mute[1].data['moments'][:,0]])
                    if file_removed:
                        print ('Need to re-run measure targets.')
                        sys.exit()

                    # check if there're NaN (it can happen when simulatin images)
                    mask= (~np.isnan(occ)) & (occ>0.) & (~np.isnan(mf)) #& (noisetier == NOISE_TIER_MASK)
                    
                    print (len(mask[mask]),len(mask))
                    # First we're going to create a table the concatenates all the current tables
                    nrows = len(noisetier[mask])
                    cols = [] 
                    cols.append(pf.Column(name="NOISETIER",format="K",array=noisetier*np.ones_like(noisetier)))#noisetier[mask]*np.ones_like(noisetier[mask])))
                    new_cols = pf.ColDefs(cols)
                    hdu = pf.BinTableHDU.from_columns(mute[1].columns + new_cols)
                    #hdu = pf.BinTableHDU.from_columns(new_cols)
                    
                    #print ('')
                    #print (len(noisetier[mask]))
                    #print (nrows)
                    #print (len(hdu.data['NOISETIER']))
                    
                    
                    
                    hdu.data['NOISETIER'] = noisetier#[mask]
                    hdu.data['NOISETIER'][~mask] = -1
                    
                    for key in ('WT_N','WT_SIG'):
                        hdu.header[key] = mute[0].header[key]
                    for cname in mute[1].columns.names:
                        sofar = 0
                        for ii, file in enumerate(files):
                            mute = pf.open(file)
                            mask =( ~np.isnan(mute[1].data['cov_even'][:,0]))& (mute[1].data['cov_even'][:,0]>0.) & ( ~np.isnan(mute[1].data['moments'][:,0]))
                            noisetier_ = np.digitize(mute[1].data['cov_even'][:,0], bins_Mf)
                            mask = mask #& (noisetier_ == NOISE_TIER_MASK)
                            nlost+=len(mask[~mask])
                            nn = len(mask)
                            hdu.data[cname][sofar:sofar+nn] = mute[1].data[cname]
                            sofar += nn  

                            
                            #nn = len(mask[mask])
                            #hdu.data[cname][sofar:sofar+nn] = mute[1].data[cname][mask]
                            #sofar += nn  
                    hdu.data['AREA'][~mask] = 1


                    for key in ('WT_N','WT_SIG'):
                        hdulist[0].header[key] = mute[0].header[key]
                    hdulist[0].header['NLOST'] = nlost
                    hdulist.append(hdu)
                    del hdu


                    # Next create an image of mean covariance matrix for each tier
                    hdr = hdulist[0].header
                    unique_tiers = np.unique(hdulist[1].data['NOISETIER'])
                    unique_tiers = unique_tiers[unique_tiers != -1]
                    
                    for tier in unique_tiers:

                        # Build covariance matrix for even parity
                        mask = hdulist[1].data['NOISETIER'] == tier
                        nn = 5
                        data = np.zeros((nn,nn))
                        index = 0
                        
                        for i in range(nn):
                            for j in range(i,nn):
                                data[i,j] = np.mean(hdulist[1].data['cov_even'][mask,index])
                                data[j,i] = np.mean(hdulist[1].data['cov_even'][mask,index])
                                index += 1
                        hdu = pf.ImageHDU(data)
                        hdu.header['WT_N'] = mute[0].header['WT_N']
                        hdu.header['WT_SIG'] = mute[0].header['WT_SIG']
                        hdu.header['TIERNAME'] = tier
                        hdu.header['TIERLOST'] = 0
                        # Record mean covariance of odd moments in header
                        cov_mean = np.mean(hdulist[1].data['cov_odd'][mask,:],axis=0)
                        mean_c = 0.5*(np.mean(hdulist[1].data['cov_odd'][mask,2])+np.mean(hdulist[1].data['cov_odd'][mask,0]))
                        hdu.header['COVMXMX'] = mean_c
                        hdu.header['COVMXMY'] = np.mean(hdulist[1].data['cov_odd'][mask,1])
                        hdu.header['COVMYMY'] = mean_c
                        hdulist.append(hdu)
                        

                        del hdu



                    try:
                        hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add))
                    except:
                        try:
                            hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add),clobber = True)# 
                        except:
                            pass





