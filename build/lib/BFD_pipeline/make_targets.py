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
make a code that assembles everything, then it must call the create_tiers from Gary.


python createTiers.py X.fits --snMin --snMax  --fluxMin --fluxMax

'''

def produce_sn(t):
    return t[1].data['moments'][:,0]/np.sqrt(t[1].data['cov_even'][:,0])

def produce_cov(t):
    return (t[1].data['cov_even'][:,0])

def produce_mf(t):
    return t[1].data['moments'][:,0]

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
        add_labels = ['','ISp_','ISm_']
        
        # make the list for simulted tiles, they need to be loaded in the same order for noise cancelling.
        files_simulated_p = glob.glob(output_folder+'/targets/{1}{0}*.fits'.format('targets','ISp_'))
        files_simulated_m = [f.replace('ISp_','ISm_') for f in files_simulated_p]      
        
      
        for add in add_labels:
            nlost = 0
            
            for xx,tt in enumerate(['targets']):
                #'''
                
                
                if not os.path.exists(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add)):
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
                                occ  = mute[1].data['cov_even'][:,0]
                                mf  = mute[1].data['moments'][:,0]
                            else:
                                occ = np.hstack([occ,mute[1].data['cov_even'][:,0]])
                                mf = np.hstack([mf,mute[1].data['moments'][:,0]])
                        if file_removed:
                            print ('Need to re-run measure targets.')
                            sys.exit()

                        # check if there're NaN (it can happen when simulatin images)
                        mask= (~np.isnan(occ)) & (occ>0.) & (~np.isnan(mf)) #& (noisetier == NOISE_TIER_MASK)

                        print (len(mask[mask]),len(mask))
                        # First we're going to create a table the concatenates all the current tables
                        cols = [] 

                        names = [mute[1].columns[i].name for i in range(len(mute[1].columns))]
                        if not ('AREA' in names):
                            cols.append(pf.Column(name="AREA",format="K",array=0*np.ones_like(mask)))#noisetier[mask]*np.ones_like(noisetier[mask])))


                        cols.append(pf.Column(name="NOISETIER",format="K",array=0*np.ones_like(mask)))#noisetier[mask]*np.ones_like(noisetier[mask])))
                        new_cols = pf.ColDefs(cols)
                        hdu = pf.BinTableHDU.from_columns(mute[1].columns + new_cols)
                        hdu.data['NOISETIER'][~mask] = -1
                        try:
                             hdu.data['AREA'][~mask] = 1
                        except:
                            pass
                        for key in ('WT_N','WT_SIG'):
                            hdu.header[key] = mute[0].header[key]
                        for cname in mute[1].columns.names:
                            sofar = 0
                            for ii, file in enumerate(files):
                                mute = pf.open(file)
                                mask =( ~np.isnan(mute[1].data['cov_even'][:,0]))& (mute[1].data['cov_even'][:,0]>0.) & ( ~np.isnan(mute[1].data['moments'][:,0]))

                                nlost+=len(mask[~mask])
                                nn = len(mask)
                                hdu.data[cname][sofar:sofar+nn] = mute[1].data[cname]
                                sofar += nn  



                        for key in ('WT_N','WT_SIG'):
                            hdulist[0].header[key] = mute[0].header[key]
                        hdulist[0].header['NLOST'] = nlost
                        hdulist.append(hdu)
                        del hdu



                        try:
                            hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add))
                        except:
                            try:
                                hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add),clobber = True)# 
                            except:
                                pass


        # MAKE NOISE TIERS *******************************************************************************
        filex = (output_folder+'/{1}{0}_sample_g.fits'.format(tt,add))
        print ('')
        print ('')
        
        if os.path.exists(output_folder+'/noisetiers.fits'):
            os.remove(output_folder+'/noisetiers.fits')
        os.system('python createTiers.py {0} --snMin {1} --snMax  {2} --fluxMin {3} --fluxMax {4} --output {5}'.format(filex,config['sn_min'],config['sn_max'],config['Mf_min'],config['Mf_max'],output_folder+'/noisetiers.fits'))
        
        # ASSIGN NOISE TIERS ********************
        from bfd import TierCollection

        tc = TierCollection.load(output_folder+'/noisetiers.fits')
        
        # here need to load it and assign noise tiers to the fits file again ---
        for add in add_labels:
            nlost = 0
            hdulist = pf.HDUList([pf.PrimaryHDU()])
            if os.path.exists(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add)):
                for xx,tt in enumerate(['targets']):
                    m_ = pf.open(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add))
                    n_ = tc.assign(m_[1].data['cov_even'])
                    noise_tiers = np.zeros(m_[1].data['cov_even'].shape[0]).astype(np.int)
                    for key in n_.keys():
                        noise_tiers[n_[key]]=key

                    # dismis stuff that doesn't pass the cuts ***********
                    
                    mask = (produce_sn(m_)>config['sn_min']) & (produce_sn(m_)< config['sn_max']) & (produce_mf(m_)>config['Mf_min']) & (produce_mf(m_)< config['Mf_max']) 

                    hdu = pf.BinTableHDU.from_columns(m_[1].columns)
                    hdu.data['NOISETIER'] = noise_tiers
                    print (len(mask[mask]),len(mask))
                    hdu.data['AREA'][~mask] = 1

                    for key in ('WT_N','WT_SIG'):
                        hdu.header[key] = m_[0].header[key]




                    for key in ('WT_N','WT_SIG'):
                        hdulist[0].header[key] = m_[0].header[key]
                    hdulist[0].header['NLOST'] = nlost
                    hdulist.append(hdu)
                    del hdu



                    try:
                        hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add))
                    except:
                        try:
                            hdulist.writeto(output_folder+'/{1}{0}_sample_g.fits'.format(tt,add),clobber = True)# 
                        except:
                            pass
          