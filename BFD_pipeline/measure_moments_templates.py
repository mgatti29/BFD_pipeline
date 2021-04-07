import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable
from .read_meds_utils import Image, MOF_table, DetectionsTable
from .utilities import update_progress, save_obj, load_obj
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmixer
import ngmix.gmix as gmix
import ngmix
from mpi4py import MPI             
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


def pipeline(config, output_folder,params_template, bands, dictionary_runs, count, MOF_deep_field,deep_fields_catalog):

        tab = TemplateTable(n = params_template['n'],
                       sigma = params_template['sigma'],
                        sn_min = 0.,#self.params['sn_min'], 
                        sigma_xy = 0.,#self.params['sigma_xy'], 
                        sigma_flux = 0.,#self.params['sigma_flux'], 
                        sigma_step = 0.,#self.params['sigma_step'], 
                        sigma_max = 0.,#self.params['sigma_max'],
                        xy_max = 0.)#self.params['xy_max'])

        tab_detections = DetectionsTable(params_template)

        included_detections = 0
        discarded_detections_n = 0
        discarded_detections_mask = 0
        count_templates = 0


        start = timeit.default_timer()
        tile = list(dictionary_runs.keys())[count]

        # Read meds and psf files for a given tile ************************************
        m_array = [meds.MEDS(dictionary_runs[tile][band]['meds'][0]) for band in bands]
        psf_array = [psfex.PSFEx(dictionary_runs[tile][band]['psf'][0])for band in bands]

        # loop on detections **********************************************************
        print ('number of images: ', (m_array[0].size))
        for index in range(m_array[0].size):
            # Initialise the detection 
            Wide_g = Image(index, meds = m_array, bands = config['bands'])
            tab_detections.add_image(Wide_g)
            
        # loads MOF **********
        tab_detections.add_MOF_models(MOF_deep_field)
        
        # this checks if anything has been dropped from the fiducial deep field catalog
        pos = np.arange(len(np.array(tab_detections.ID_array)))
        df1 = pd.DataFrame(data = {'pos': pos} , index = tab_detections.ID_array)
        df2 = pd.DataFrame(index = deep_fields_catalog.index)        
        uu = df2.join(df1)
        uu = uu.dropna()
        discard = pos[~np.in1d(pos,(uu['pos']))]
        for d in discard:
            if (tab_detections.images[d].flags == 0):
                tab_detections.images[d].flag = 10
                
        total_ob = np.min([config['max_target_per_tile'],m_array[0].size])
        start = timeit.default_timer()
        for index in range(total_ob):
            update_progress((index*1./total_ob),timeit.default_timer(),start)
                    
            tab_detections.images[index].Load_MEDS(index, meds = m_array)   
            # check masked bands and only consider objects with 2+ bands not masked
            bands_not_masked = tab_detections.images[index].check_mfrac(limit=0.1, use_COADD_only = True)
            #subtract background
            tab_detections.images[index].subtract_background()
            #computes noise based on mask and weightmap, and then discard them.
            tab_detections.images[index].compute_noise()
            
            tab_detections.images[index].add_PSF_model(index, psf = psf_array, psf_type = 'PSFex', bands = bands)
            tab_detections.images[index].zero_padd_psf()
            
                    
            # cut the psf stamp - for testing mostly
            if config['cut_psf'][0]:
                for b in range(len(tab_detections.images[index].bands)):
                    for i in range((tab_detections.images[index].ncutout[b])):
                        NN = tab_detections.images[index].psf[b][i].shape[0]
                        M = np.zeros((NN,NN))
                        u = np.int((NN-config['cut_psf'][1])/2)
                        M[u:-u,u:-u] = 1.
                        tab_detections.images[index].psf[b][i] *= M
                                
            if len(bands_not_masked) < 2:
                #print ('too many pixels masked')
                tab_detections.images[index].flags += 1
                
                # the image is not usable, so discard the images
                tab_detections.images[index].make_WCS() #the image might be still used for mof subtraction so we need the wcs
                tab_detections.images[index].imlist = None
                tab_detections.images[index].seglist = None
                tab_detections.images[index].masklist = None
                tab_detections.images[index].wtlist = None
                tab_detections.images[index].psf = None
            else:
                    
                # read WCS and compute moments
                tab_detections.images[index].make_WCS()

                tab_detections.render_MOF_models(index = index, render_self = False, render_others = True)
                mute_range = [index,index+1]
        
                tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = True, flags = 0, MOF_subtraction = config['MOF_subtraction'], band_dict = params_template['band_dict'], chunk_range = mute_range)
                tab_detections.images[index].imlist = None
                tab_detections.images[index].seglist = None
                tab_detections.images[index].masklist = None
                tab_detections.images[index].wtlist = None
                tab_detections.images[index].jaclist = None
                tab_detections.images[index].psf = None
                tab_detections.images[index].MOF_model_rendered = None
                gc.collect()

            #
        del MOF_deep_field
        gc.collect()
        path = output_folder+'/templates/'+'/templates_'+tile
        
        save_obj(path,tab_detections)
        




def measure_moments_templates(output_folder,**config):
    '''
    It computes the moments from des y6 deef fields tiles.
    '''
    
    # Read the config file
    print ('Executing the measure_moments_templates stage')
    
    '''
    This reads the fiducial deep field catalog. It is needed as we need to match the ra,dec from this catalog with the ra,dec from the 
    detections contained in the single-exposure meds files.
    '''
    
    deep_fields_catalog_u = pf.open(config['deep_fields_catalog'])
    deep_fields_catalog = pd.DataFrame.from_dict({'ra':np.array(deep_fields_catalog_u[1].data['RA']).byteswap().newbyteorder(),
                                                  'dec':np.array(deep_fields_catalog_u[1].data['DEC']).byteswap().newbyteorder(),
                                                  'TILENAME':np.array(deep_fields_catalog_u[1].data['TILENAME']).byteswap().newbyteorder(),
                                                  'ID':deep_fields_catalog_u[1].data['ID']})
    
    
    


    # config for computation of moments
    params_template = {}
    params_template['n'] = config['n'] # 4 default. KSigmaWeight function index   
    params_template['sigma'] = config['sigma'] # sigma KSigmaWeight         
    mute = dict()
    for i in range(len(config['band_dict'])):
        mute[config['band_dict'][i][0]] = bfd.BandInfo(config['band_dict'][i][1],i)
    params_template['band_dict'] = mute
    params_template['bands'] = config['bands']
    

    # read MOF deep fields tiles
    MOF_deep_field = MOF_table(config['MOF_table_path'])
    deep_fields_coadd_path = config['path_coadd_deepfields']
    fields = config['fields']
    bands = config['bands']

    
    dictionary_runs = dict()
    for field in fields:
        folders = glob.glob(deep_fields_coadd_path+field+'/*/')
        for folder in folders:
            tile = (folder.split(deep_fields_coadd_path+field+'/')[1]).split('/')[0]
            dictionary_runs[tile] = dict()
            for band in bands:
                dictionary_runs[tile][band] = dict()
                dictionary_runs[tile][band]['meds'] = glob.glob(folder+'/*{0}_meds*'.format(band))
                dictionary_runs[tile][band]['psf'] = glob.glob(folder+'/*{0}_psfcat*'.format(band))
    
    print ('Number of tiles: ',len(dictionary_runs.keys()))
    
    #check if it has already been run:
    if not os.path.exists(output_folder+'/templates/'):
        os.mkdir(output_folder+'/templates/')
    mask = np.array([not os.path.exists(output_folder+'/templates/'+'/templates_'+ff+'.pkl') for ff in list(dictionary_runs.keys())])
        

    run_count = 0
    list_run= np.arange(len(list(dictionary_runs.keys())))[mask]
    print ('Number of tiles to be run : ',len(list_run))

    while run_count<len(list_run):
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if run_count+comm.rank<len(list_run):
            pipeline(config, output_folder, params_template, bands, dictionary_runs, list_run[(run_count+comm.rank)],MOF_deep_field,deep_fields_catalog)
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 

        
        