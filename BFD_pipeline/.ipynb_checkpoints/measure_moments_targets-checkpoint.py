
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


def measure_moments_targets(output_folder,**config):
    '''
    It computes the moments from des y6 tiles.
    '''
    
    # Read the config file
    print ('Executing the measure_target_moments stage')
    
    #tiles_available = np.load(config['path_data']+'tiles.npy')
    
    for i, b in enumerate(config['bands']):
        files = glob.glob(config['path_data']+str(b)+'/*fz') 
        if i == 0:
            tiles = ['DES'+f.split('DES')[1].split('_')[0] for f in files]
        else:
            tiles1 = ['DES'+f.split('DES')[1].split('_')[0] for f in files]
            tiles = np.hstack([tiles,tiles1])
        tiles_available = np.unique(tiles)
    
    print ('Number of tiles available: ',len(tiles_available))

    
    if config['tiles'] == 'All':
        tiles_to_be_used = tiles_available
    elif type(config['tiles']) is list:
        mask = np.in1d(tiles_available,config['tiles'])
        tiles_to_be_used = tiles_available[mask]
    elif type(config['tiles']) is int:
        # randomly select X tiles
        index = np.random.randint(0,len(tiles_available),config['tiles'])
        tiles_to_be_used = tiles_available[index]
        
    print ('Selected ',len(tiles_to_be_used),' tile(s).')
    
    
    # makes a dictionary of the tiles ****
    dictionary_runs = dict()
    for  tile in tiles_to_be_used:
        dictionary_runs[tile] = dict()
        for band in config['bands']:
            dictionary_runs[tile][band] = dict()
            dictionary_runs[tile][band]['meds'] = glob.glob(config['path_data']+band+'/'+tile+'*fits.fz')[0]
             
                
    # list of the runs to do. 
    list_run = []
    for count, tile in enumerate(dictionary_runs.keys()):
        if not os.path.exists(output_folder+'/targets/'):
            os.mkdir(output_folder+'/targets/')
        path = output_folder+'/targets/targets_'+tile
        if not os.path.exists(path+'_chunk_last'+'.pkl'):
            list_run.append(count)
    print ('Runs to do: ',len(list_run))
    config['output_folder'] = output_folder
    run_count = 0
    while run_count<len(list_run):
        comm = MPI.COMM_WORLD
        #try:
        if (run_count+comm.rank) < len(list_run):
            pipeline(config, dictionary_runs, list_run[run_count+comm.rank])
        else:
            index = np.int((run_count+comm.rank) /len(list_run))
            pipeline(config, dictionary_runs, list_run[run_count+comm.rank-len(list_run)*index-index])
        #except:
        #    pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
            
def pipeline(config, dictionary_runs, count):
        # define the tile ***
        tile = list(dictionary_runs.keys())[count]
        print ('TILE running: ',tile)
        path = config['output_folder']+'/targets/targets_'+tile
        params_template = dict()
        params_template['n'] = config['n'] # 4 default. KSigmaWeight function index   
        params_template['sigma'] = config['sigma'] # sigma KSigmaWeight         
        
        mute = dict()
        for i in range(len(config['band_dict'])):
            mute[config['band_dict'][i][0]] = bfd.BandInfo(config['band_dict'][i][1],i)
        params_template['band_dict'] = mute
        params_template['bands'] = config['bands']
    
        tab_detections = DetectionsTable(params_template)

        start = timeit.default_timer()

        # Read meds and psf files for a given tile ************************************
        m_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands']]

        print ('number of images: ', (m_array[0].size))
        for index in range(m_array[0].size):
            # Initialise the detection 
            Wide_g = Image(index, meds = m_array, bands = config['bands'])
            tab_detections.add_image(Wide_g)

        # loads MOF **********
        files = []
        mute = glob.glob(config['path_data']+'mof/'+tile+'*')
        if len(mute)>0:
            for m in mute:
                files.append(m)
        print ('Loading MOF solutions')

        MOF_wide_field = MOF_table(files)
        try:
            tab_detections.add_MOF_models(MOF_wide_field)
        except:
            print ('failed to add MOF')
        print ('loading images and computing moments')
        
        # division in chunks *********
        
        len_file = np.min([len(tab_detections.images),config['max_target_per_tile']])
        chunk_size = config['chunk_size']
        
        pool = multiprocessing.Pool(processes=config['agents_chunk'])

        runs = math.ceil(len_file/chunk_size)
        xlist = range(runs)
        print ('subruns for this tile: ',runs,' chunk size: ', chunk_size)
        _ = pool.map(partial(f, config = config, params_template = params_template,chunk_size=chunk_size, path = path, tab_detections = tab_detections, m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs), xlist)
        #for x in xlist:
        #    f(x, config = config,params_template = params_template, chunk_size=chunk_size, path = path, tab_detections = tab_detections, m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs)
        
        
def f(iii, config, params_template, chunk_size, path, tab_detections, m_array, bands, len_file, runs):
            m_array = [meds.MEDS(m_array[band]['meds']) for band in bands]
            chunk_range =  [int(chunk_size*iii),int(np.min([chunk_size*(iii+1),len_file]))] 
            start = timeit.default_timer()
            if iii == (runs)-1:
                path_save1 = path+'_chunk_last'.format(iii)
            else:
                path_save1 = path+'_chunk_{0}'.format(iii)

            if not os.path.exists(path_save1+'.pkl'):
                print (path_save1)
                for index in range(chunk_range[0],chunk_range[1]):
                    update_progress((index*1.+1-chunk_range[0])/(chunk_range[1]-chunk_range[0]),timeit.default_timer(),start)
                    # load meds for image 'index'
                    tab_detections.images[index].Load_MEDS(index, meds = m_array)
                    
                    #setup flag
                    tab_detections.images[index].flags = 0
                    # check masked bands and only consider objects with 2+ bands not masked
                    bands_not_masked = tab_detections.images[index].check_mfrac(limit=0.1, use_COADD_only = True)
                    #subtract background
                    tab_detections.images[index].subtract_background()
                    #computes noise based on mask and weightmap, and then discard them.
                    tab_detections.images[index].compute_noise()
                    
                    # cut the psf stamp - for testing mostly
                    if config['cut_psf'][0]:
                        for b in range(len(tab_detections.images[index].bands)):
                            for i in range((tab_detections.images[index].ncutout[b])):
                                M = np.zeros((tab_detections.images[index].psf[b][i].shape[0],tab_detections.images[index].psf[b][i].shape[1]))
                                u = config['cut_psf'][1]
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
                        tab_detections.images[index].zero_padd_psf()
                        # read WCS and compute moments
                        tab_detections.images[index].make_WCS()
                        tab_detections.images[index].compute_psf_fwhm(use_COADD_only = True)
                        tab_detections.images[index].compute_psf_params(use_COADD_only = True, band_dict = params_template['band_dict'])
                        
                        
                    
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
          
                if iii == (runs)-1:
                    save_obj(path+'_chunk_last'.format(iii),tab_detections)

                else:
                    save_obj(path+'_chunk_{0}'.format(iii),tab_detections)