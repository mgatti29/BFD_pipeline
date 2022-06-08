import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
from .read_meds_utils import Image, MOF_table, DetectionsTable,BandInfo

from .utilities import save_obj, load_obj
import frogress
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
import ngmix
import galsim
import joblib
from .mcal_routines import *
import metacal_m
from metacal_m import MetacalFitter, CONFIG
from ngmix.jacobian.jacobian_nb import jacobian_get_vu, jacobian_get_area

from bfd.momentcalc import MomentCovariance

def save_(self,fitsname,config):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        try:
            col.append(fits.Column(name="id_simulated_gal",format="K",array=self.p0))
            col.append(fits.Column(name="id_simulated_PSF",format="K",array=self.p0_PSF))
        except:
            pass
        
        col.append(fits.Column(name="Mf_per_band",format="{0}E".format(np.array(self.meb).shape[1]),array=self.meb))
        try:
            col.append(fits.Column(name="true_fluxes",format="{0}E".format(np.array(self.meb).shape[1]),array=self.true_fluxes))
            
        except:
            pass
        
        col.append(fits.Column(name="moments",format="5E",array=self.moment))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))
        
        col.append(fits.Column(name="ra",format="D",array=self.ra))
        col.append(fits.Column(name="dec",format="D",array=self.dec))
        
        PSF_moments = np.vstack([np.array(self.psf_Mf),np.array(self.psf_Mr),np.array(self.psf_M1),np.array(self.psf_M2)]).T
        col.append(fits.Column(name="psf_moments",format="4E",array=PSF_moments))

        try:
            col.append(fits.Column(name="w_i",format="D",array=self.band1))
            col.append(fits.Column(name="w_r",format="D",array=self.band2))
            col.append(fits.Column(name="w_z",format="D",array=self.band3))
        except:
            pass
        

            
            
            
            
            

        try:
            col.append(fits.Column(name="g1_mcal",format="E",array= np.array(self.g1_mcal)))
            col.append(fits.Column(name="g2_mcal",format="E",array= np.array(self.g2_mcal)))
            col.append(fits.Column(name="g1p_mcal",format="E",array= np.array(self.g1p_mcal)))
            col.append(fits.Column(name="g2p_mcal",format="E",array= np.array(self.g2p_mcal)))
            col.append(fits.Column(name="g1m_mcal",format="E",array= np.array(self.g1m_mcal)))
            col.append(fits.Column(name="g2m_mcal",format="E",array= np.array(self.g2m_mcal)))
            col.append(fits.Column(name="mcal_sn",format="E",array= np.array(self.mcal_sn)))
            col.append(fits.Column(name="mcal_sn_1p",format="E",array= np.array(self.mcal_sn_1p)))
            col.append(fits.Column(name="mcal_sn_2p",format="E",array= np.array(self.mcal_sn_2p)))
            col.append(fits.Column(name="mcal_sn_1m",format="E",array= np.array(self.mcal_sn_1m)))
            col.append(fits.Column(name="mcal_sn_2m",format="E",array= np.array(self.mcal_sn_2m)))
            col.append(fits.Column(name="mcal_flags",format="E",array= np.array(self.mcal_flags)))
            col.append(fits.Column(name="mcal_size_ratio",format="E",array= np.array(self.mcal_size_ratio)))
            col.append(fits.Column(name="mcal_size_ratio_1p",format="E",array= np.array(self.mcal_size_ratio_1p)))
            col.append(fits.Column(name="mcal_size_ratio_2p",format="E",array= np.array(self.mcal_size_ratio_2p)))
            col.append(fits.Column(name="mcal_size_ratio_1m",format="E",array= np.array(self.mcal_size_ratio_1m)))
            col.append(fits.Column(name="mcal_size_ratio_2m",format="E",array= np.array(self.mcal_size_ratio_2m)))
            
        except:
            pass

        try:
            l = np.array(self.meb).shape[1]
            col.append(fits.Column(name="cov_Mf_per_band",format="{0}E".format(l),array=np.array(self.cov_even_per_band)[:,0,:]))

            
    
        except:
            pass
        
        try:
            col.append(fits.Column(name="des_id",format="D",array=self.des_id))
            col.append(fits.Column(name="photoz",format="D",array=self.photoz))
        except:
            pass

        #col.append(fits.Column(name="NOISETIER",format="K",array=noisetier*np.ones_like(self.id)))
        
        if len(self.num_exp) == len(self.id):
            col.append(fits.Column(name="num_exp",format="K",array=self.num_exp))
        #if self.cov_even is not None:
        #    # saved in order M0xM0, M0xMR, M0xM1, M0xM2, M0xMC,
        #    # MRxMR, MRxM1, MRxM2, MRxMC, M1xM1, M1xM2, M1xMC,
        #    # M2xM2, M2xMC, MCxMC (total of 15)
        #    col.append(fits.Column(name="cov_even",format="15E",array=self.cov_even))
        #    # saved in order MXxMX, MXxMY, MYxMY (total of 3)
        #    col.append(fits.Column(name="cov_odd",format="3E",array=self.cov_odd))
            
            
        #if len(self.delta_flux_moment) == len(self.id):
        #    col.append(fits.Column(name="delta_flux_moment",format="E",array=self.delta_flux_moment))
        #if len(self.cov_delta_flux_moment) == len(self.id):
        #    col.append(fits.Column(name="cov_delta_flux_moment",format="E",array=self.cov_delta_flux_moment))
            
        
        #print (np.array(self.cov).shape)
        #print (MomentCovariance.unpack(np.array(self.cov)))
        #print (MomentCovariance.unpack(np.array(self.cov)).odd)
        col.append(fits.Column(name="covariance",format="15E",array=np.array(self.cov).astype(np.float32)))
        #if self.area is not None:
        #    col.append(fits.Column(name="area",format="E",array=self.cov.astype(np.float32)))
        #if self.select is not None:
        #    col.append(fits.Column(name="select",format="I",array=self.cov.astype(np.int16)))
        #    


            

        #self.prihdu.header['NLOST'] = self.nlost  # Update value
        if config['setup_image_sims']:
            self.prihdu.header['STAMPS'] = 1  # Update value
            col.append(fits.Column(name="AREA",format="K",array=np.zeros(len(self.id))))
        
        else:
            self.prihdu.header['STAMPS'] = 0
            #print ('AREA ', self.area)
            col.append(fits.Column(name="AREA",format="K",array=self.area))
            
        cols=fits.ColDefs(col)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
      
        return
    
def measure_moments_targets(output_folder,**config):
    '''
    It computes the moments from des y6 tiles.
    '''
    
    # Read the config file
    print ('Executing the measure_target_moments stage')
    if config['MPI']:
        from mpi4py import MPI 
    # this checks how many tiles can be used. ****************************************
    for i, b in enumerate(config['bands']):
        files = glob.glob(config['path_data']+str(b)+'/*fz') 
        if i == 0:
            tiles = ['DES'+f.split('DES')[1].split('_')[0] for f in files]
        else:
            tiles1 = ['DES'+f.split('DES')[1].split('_')[0] for f in files]
            tiles = np.hstack([tiles,tiles1])
        tiles_available = np.unique(tiles)
    
    print ('Number of tiles available: ',len(tiles_available))

    # this selects the tiles to be used based on the config entries *******************
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
    
    
    # makes a dictionary of the tiles *************************************************
    dictionary_runs = dict()
    for tile in tiles_to_be_used:
        dictionary_runs[tile] = dict()
        for band in config['bands']:
            dictionary_runs[tile][band] = dict()
            dictionary_runs[tile][band]['meds'] = glob.glob(config['path_data']+band+'/'+tile+'*fits.fz')[0]
             
                
    # list of the runs to do. ********************************************************
    list_run = []
    for count, tile in enumerate(dictionary_runs.keys()):
        if not os.path.exists(output_folder+'/targets/'):
            try:
                os.mkdir(output_folder+'/targets/')
            except:
                pass
        list_run.append(count)
    print ('Runs to do: ',len(list_run))
    config['output_folder'] = output_folder
    run_count = 0

    
    # Runs the main pipeline **********************************************************
    if ((config['MPI_per_tile']) or (not config['MPI'])):
        while run_count<len(list_run):
            pipeline(config, dictionary_runs, list_run[run_count])
            run_count+=1
    else:
        while run_count<len(list_run):
            comm = MPI.COMM_WORLD
      
            if (run_count+comm.rank) < len(list_run):
                pipeline(config, dictionary_runs, list_run[run_count+comm.rank])

            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
        
            
def pipeline(config, dictionary_runs, count):
        if config['MPI']:
            from mpi4py import MPI 
        tile = list(dictionary_runs.keys())[count]
        print ('TILE running: ',tile)
        
        if config['setup_image_sims']:
            path = config['output_folder']+'/targets/ISp_targets_'+tile
        else:
            path = config['output_folder']+'/targets/targets_'+tile
            
        params_template = dict()
        params_template['n'] = config['n'] # 4 default. KSigmaWeight function index   
        params_template['sigma'] = config['sigma'] # sigma KSigmaWeight         
        
        mute_b = dict()
        for i in range(len(config['band_dict'])):
            mute_b[config['band_dict'][i][0]] = config['band_dict'][i][1]
        #params_template['band_dict'] = mute
        params_template['bands'] = config['bands']
        params_template['band_dict'] = dict()
        params_template['band_dict']['bands'] = list(config['bands'])
        w = []
        for b in config['bands']:
            w.append(mute_b[b])
        params_template['band_dict']['weights'] = list(w)
        params_template['band_dict']['index'] = list(np.arange(len(w)))

        tab_detections = DetectionsTable(params_template)

        start = timeit.default_timer()

        # Read meds and psf files for a given tile ************************************
        m_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands']]

       
        print ('number of images: ', (m_array[0].size))
        for index in range(m_array[0].size):
            # Initialise the detection 
            Wide_g = Image(index, meds = m_array, bands = config['bands'])
            tab_detections.add_image(copy.deepcopy(Wide_g))
        del m_array
        gc.collect()

        # loads MOF **********
        files = []
        if not config['shredder_x']:
            mute = glob.glob(config['path_data']+'mof/'+tile+'*')
        else:
            mute = glob.glob(config['path_data']+'shredex/'+tile+'*')
            
        if len(mute)>0:
            for m in mute:
                files.append(m)
        print ('Loading MOF solutions')

        
        #try:
        if 1==1:
            MOF_wide_field = MOF_table(files[0],config['shredder'])
            tab_detections.add_MOF_models(MOF_wide_field)
            # save the MOF models --
            if not os.path.exists(config['output_folder']+'/MOF_models/'):
                try:
                    os.mkdir(config['output_folder']+'/MOF_models/')
                    
                except:
                    pass
            p_ = config['output_folder']+'/MOF_models/{0}_{1}.npy'.format(tile,len(tab_detections.images)-1)
            if not os.path.exists(p_):
                print ('pre-saving MOF fits')
                for ii_, index in enumerate(range(len(tab_detections.images))):
                    #try:
                        np.save(config['output_folder']+'/MOF_models/{0}_{1}'.format(tile,index),tab_detections.images[index].MOF_models)
                        del tab_detections.images[index].MOF_models
                        if ii_% 1000 == 0:
                            gc.collect()
                #except:
                #    pass
                    
            
        #except:
        #    print ('failed to add MOF')
        print ('loading images and computing moments')
        
        # if it's an image sims run, load the MOF parameters for PSF and galaxies.
        params_image_sims = dict()
        if config['setup_image_sims']:
            params_image_sims = np.load(config['simulated_templates'],allow_pickle='TRUE').item()
        # division in chunks *********
        len_file = np.min([len(tab_detections.images),config['max_target_per_tile']])
        external = False
        if config['external']:
            external = np.load(config['external_path'],allow_pickle=True).item()
            len_file = len(external['ra'])
            
        chunk_size = config['chunk_size']
    
        runs = math.ceil(len_file/chunk_size)
        xlist = range(runs)
        print ('subruns for this tile: ',runs,' chunk size: ', chunk_size)
        
       
        if config['MPI_per_tile']:
            run_count = 0
            
            while run_count<runs:
                comm = MPI.COMM_WORLD
                if run_count+comm.rank<runs:
        
                    f(run_count+comm.rank, config = config, params_template = params_template,chunk_size=chunk_size, path = path, tab_detections =  copy.deepcopy(tab_detections), m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs,params_image_sims = params_image_sims,external=external,tile=tile)
                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        else:
            if config['agents_chunk'] > 1:
                pool = multiprocessing.Pool(processes=config['agents_chunk'])

                _ = pool.map(partial(f, config = config, params_template = params_template,chunk_size=chunk_size, path = path, tab_detections =  copy.deepcopy(tab_detections), m_array = copy.deepcopy(dictionary_runs[tile]), bands = config['bands'], len_file = len_file, runs = runs,params_image_sims = params_image_sims,external=external,tile=tile), xlist)
            else:
                for x in xlist:
                    f(x, config = config, params_template = params_template,chunk_size=chunk_size, path = path, tab_detections =  copy.deepcopy(tab_detections), m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs, params_image_sims = params_image_sims,external=external,tile=tile)

def f(iii, config, params_template, chunk_size, path, tab_detections, m_array, bands, len_file, runs, params_image_sims,external,tile):
            
            # inititalise target table ---------
            tab_targets = TargetTable(n = params_template['n'],
                                  sigma = params_template['sigma'])

            tab_targets.ra = []
            tab_targets.dec = []
            tab_targets.psf_Mf = []
            tab_targets.psf_Mr = []
            tab_targets.psf_M1= []
            tab_targets.psf_M2= []
            tab_targets.area= []
            
            
            if config['setup_image_sims']:
                
                tab_targets.g1_mcal= []
                tab_targets.g2_mcal= []
                tab_targets.g1m_mcal= []
                tab_targets.g2m_mcal= []
                tab_targets.g1p_mcal= []
                tab_targets.g2p_mcal= []
                tab_targets.mcal_sn= []
                tab_targets.mcal_flags= []
                tab_targets.mcal_size_ratio= []



                tab_targets.mcal_sn_1p= []
                tab_targets.mcal_sn_1m= []
                tab_targets.mcal_sn_2p= []
                tab_targets.mcal_sn_2m= []

                tab_targets.mcal_size_ratio_1p = []
                tab_targets.mcal_size_ratio_1m = []
                tab_targets.mcal_size_ratio_2p = []
                tab_targets.mcal_size_ratio_2m = [] 

            tab_targets.band1 = []
            tab_targets.band2 = []
            tab_targets.band3 = []
            tab_targets.p0 = []
            tab_targets.p0_PSF = []
            tab_targets.meb = []
            tab_targets.true_fluxes = []
            tab_targets.cov_odd_per_band = []
            tab_targets.cov_even_per_band = []
                    
            
            if config['setup_image_sims']:
                
                tab_targets.des_id = []
                tab_targets.photoz = []
                tab_targets_m = copy.deepcopy(tab_targets)
            
            m_array = [meds.MEDS(m_array[band]['meds']) for band in bands]
            chunk_range =  [int(chunk_size*iii),int(np.min([chunk_size*(iii+1),len_file]))] 
            start = timeit.default_timer()
            path_save1 = path+'_chunk_{0}'.format(iii)
            
            
            # try to read if we want to use a particular stamp for the simulations ---
            try:
                index_fixed = copy.copy(config['index_fixed'])
            except:
                index_fixed = False
                
            # set them to 0 in case it's the run on data ----
            p0 = 0
            p0_PSF = 0
            res_p = []
            res_m = []    

                
            count = 0
            if config['setup_image_sims']:
  
                    parhx= (path+'_chunk_{0}.fits'.format(iii)).replace('ISp','ISm')
            else:
                    parhx = path+'_chunk_{0}.fits'.format(iii)
        
            #print (chunk_range,len_file)
            
            if not os.path.exists(parhx):
        
                #for index_t in frogress.bar(range(chunk_range[0])):
                #    tab_detections.images[index_t].imlist = None
                #    tab_detections.images[index_t].seglist = None
                #    tab_detections.images[index_t].masklist = None
                #    tab_detections.images[index_t].wtlist = None
                #    #tab_detections.images[index_t].jaclist = None
                #    tab_detections.images[index_t].psf = None
                    
                '''
                For the selection term, I need to generate a stamp from the weight map
                
                tab_detections.images[index].Load_MEDS(index, meds = m_array)
                tab_detections.images[index].make_WCS()
                tab_detections.images[index].make_false_stamp(use_COADD_only=config['use_COADD_only'])
                

                
                
                '''

                if config['MOF_subtraction']:
                    print ('pre checks MOF subtraction')
                    idx_n_array = []
                    for index in frogress.bar(range(chunk_range[0],chunk_range[1])):
                        
                        
                        
                        tab_detections.images[index].Load_MEDS_fast(index, meds = m_array)
                        for b in range(len(tab_detections.images[index].seglist)):
                            idx_n = np.unique(tab_detections.images[index].seglist[b][0].flatten())
                            idx_n = idx_n[(idx_n!=0)&(idx_n!=index+1)]#&(~np.in1d(idx_n,np.array(idx_n_array)))]
                            for i_ in idx_n:
                                idx_n_array.append(i_-1)

                        del tab_detections.images[index].jaclist
                        del tab_detections.images[index].seglist
                        del tab_detections.images[index].orig_start_rowcol
                        del tab_detections.images[index].orig_rowcol
                        if (index%50 == 0):
                            gc.collect()
                            
                    idx_n_array = np.unique(np.array(idx_n_array))
                    #for index in  frogress.bar(range(len(tab_detections.images))):

                        
                    for i,index in enumerate(idx_n_array):

                
                            tab_detections.images[index].Load_MEDS_fast(index, meds = m_array,load_seglist=False)
                            tab_detections.images[index].make_WCS()
                            del tab_detections.images[index].jaclist
                            
                            tab_detections.images[index].MOF_models =  np.load(config['output_folder']+'/MOF_models/{0}_{1}.npy'.format(tile,index),allow_pickle=True)
                            if (i%100 == 0):
                                gc.collect()
                        #else:
                            #try:
                            #    del tab_detections.images[index].MOF_model_rendered
                            #except:
                            #    print ('missing MODEL ',index)
                        #if (index%2000 == 0):
                        #    gc.collect()
                    print ('done pre checks')
            
            
            
            
                        
                        
                image_storage = dict()
                        
                for index_t in frogress.bar(range(chunk_range[0],chunk_range[1])):
                    try:
                        if index_fixed:
                            index = copy.copy(index_fixed)
                            if config['external']:
                                config['g1'][0] = external['e1'][index_t]
                                config['g1'][1] = -external['e1'][index_t]
                                config['g2'][0] = external['e2'][index_t]
                                config['g2'][1] = -external['e2'][index_t]
                        else:
                            index = copy.copy(index_t)
                                #'''  --- if external, index % len(tab_detections.images), but save the index for ID later '''
                                #'''  --- if external, set config[g1,g2] to external '''

                            if config['external']:
                                config['g1'][0] = external['e1'][index_t]
                                config['g1'][1] = -external['e1'][index_t]
                                config['g2'][0] = external['e2'][index_t]
                                config['g2'][1] = -external['e2'][index_t]

                                index = index_ext % len(tab_detections.images)



                        try:
                            tab_detections.images[index].Load_MEDS(index, meds = m_array)
                        except:

                            print (index,len(tab_detections.images),'   ')
                        # read WCS 
                        tab_detections.images[index].make_WCS()
                        #setup flag
                        tab_detections.images[index].flags = 0
                        # check masked bands and only consider objects with 2+ bands not masked
                        bands_not_masked = tab_detections.images[index].check_mfrac(limit=config['frac_limit'], use_COADD_only = config['COADD_only'])


                        number_of_replicas = 1
                        if config['setup_image_sims']:
                            # the 2* is to allow positively and negatively sheared simulated images.
                            number_of_replicas = 2*config['number_of_replicas']

                        # the following loop allows to use simulate images instead real galaxyes
                            #try:
                            if 1==1: 
                                if len(config['noise_ext'])==3:
                                    noise_ext = np.random.normal(config['noise_ext'][0],config['noise_ext'][1])
                                    if noise_ext<= 0:
                                        noise_ext = config['noise_ext'][0]

                                elif len(config['noise_ext'])==2:
                                    noise_ext = config['noise_ext'][0] + np.random.randint(0,10000,1)*0.0001*(config['noise_ext'][1]-config['noise_ext'][0])
                                else:
                                    noise_ext = config['noise_ext'][0]
                            #except:
                            #    noise_ext = False





                        count_replica=0
                        for replica in range(number_of_replicas):
                           # print (replica)
                            if config['setup_image_sims']:

                                if replica%2 == 0:
                                    # generate a simulated image
                                    redoit = True
                                    while redoit:

                                        sim_p,sim_m,sim_PSF,p0,p0_PSF,fluxes,jac_ = tab_detections.generate_simulated_images(index,config,params_image_sims,use_COADD_only=config['COADD_only'],noiseless=config['noiseless'],  maskless = config['maskless'],size_stamp='auto',index_P0= config['index_P0'],index_P0_PSF=config['index_P0_PSF'],noise_factor = config['noise_factor'],count_replica=count_replica ,noise_ext = noise_ext,g1=config['g1'],g2=config['g2'])
                                        count_replica+=1
                                        if sim_p is not None:
                                            redoit = False
                                    resize_sn = 1.
                                    #rng = np.random.RandomState(seed=1)
                                    #seed = rng.randint(size=1, low=1, high=2**29)[0]
                                    seed = np.random.randint(size=1, low=1, high=2**29)[0]
                                    if config['run_mcal']:
                                        if config['index_P0_PSF']=='turb':
                                             res_p_, res_m_ = run_single_sim_pair3(seed,params_image_sims[p0],p0_PSF,resize_sn,config['noise_factor']*noise_ext,config['g1'],config['g2'],turb=True)

                                        else:
                                            res_p_, res_m_ = run_single_sim_pair3(seed,params_image_sims[p0],params_image_sims[p0_PSF],resize_sn,config['noise_factor']*noise_ext,config['g1'],config['g2'])

                                    tab_detections.images[index].imlist = copy.deepcopy(sim_p)
                                    tab_detections.images[index].psf = copy.deepcopy(sim_PSF)
                                if replica%2 == 1 :
                                    tab_detections.images[index].imlist = copy.deepcopy(sim_m)
                                    tab_detections.images[index].psf = copy.deepcopy(sim_PSF)

                                # measure mcal moments *****************************


                                if config['run_mcal']:
                                    if replica%2 == 1:
                                        res_x = copy.deepcopy(res_m_)
                                    else:
                                        res_x = copy.deepcopy(res_p_)






                            if config['interp_masking']:
                                #Interpolates ther images over masked pixels
                                #print (tab_detections.images[index].imlist)
                                tab_detections.images[index].deal_with_bmask(use_COADD_only = config['COADD_only'])

                            # remove exposures that don't pass the mask fraction.

                            #subtract background
                            #tab_detections.images[index].subtract_background()
                            #computes noise based on mask and weightmap, and then discard them.
                            if config['setup_image_sims']:
                                tab_detections.images[index].compute_noise(config['noise_factor'],noise_ext)
                            else:
                                tab_detections.images[index].compute_noise()

                            # cut the psf stamp - for testing mostly
                            if config['cut_psf'][0]:
                                for b in range(len(tab_detections.images[index].bands)):
                                    for i in range((tab_detections.images[index].ncutout[b])):
                                        NN = tab_detections.images[index].psf[b][i].shape[0]
                                        M = np.zeros((NN,NN))
                                        u = np.int((NN-config['cut_psf'][1])/2)
                                        if u == 0:
                                            M = 1.
                                        else:
                                            M[u:-u,u:-u] = 1.

                                        tab_detections.images[index].psf[b][i] *= M
                            tab_detections.images[index].zero_padd_psf()


                            if len(bands_not_masked) < config['minimum_number_of_bands']:
                                #print ('too many pixels masked')
                                tab_detections.images[index].flags = 1
                            else:
                                #print ('INDEX ',index,bands_not_masked,tab_detections.images[index].mfrac[0][0],tab_detections.images[index].mfrac[1][0],tab_detections.images[index].mfrac[2][0])
                                if config['MOF_subtraction']:
                                    tab_detections.render_MOF_models(index = index, render_self = False, render_others = True,use_COADD_only = config['COADD_only'])
                                mute_range = [index,index+1]

                                image_storage[index] = dict()


                                '''
                                for index_band in range(3):
                                    start = 1
                                    end = tab_detections.images[index].ncutout[index_band]
                                    image_storage[index][index_band] = dict()
                                    for exp in range(start, end):  


                                        image_storage[index][index_band][exp] = [tab_detections.images[index].imlist[index_band][exp],tab_detections.images[index].MOF_model_rendered[index_band][exp],tab_detections.images[index].seglist[index_band][exp]]
                                '''








                                tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = config['COADD_only'], flags = 0, MOF_subtraction = config['MOF_subtraction'], band_dict = params_template['band_dict'], chunk_range = mute_range,pad_factor=config['pad_factor'])
                                del tab_detections.images[index].MOF_model_rendered
                                del tab_detections.images[index].imlist
                                gc.collect()




                                newcent=np.array([tab_detections.images[index].image_ra,tab_detections.images[index].image_dec])[:,0]+tab_detections.images[index].xyshift/3600.0

                                mom, meb = tab_detections.images[index].moments.get_moment(0.,0.,returnbands=True)

                                covm_even,covm_odd,covm_even_all,covm_odd_all = tab_detections.images[index].moments.get_covariance(returnbands=True)

                                covgal = covm_even,covm_odd
                                covgal_per_band = covm_even_all,covm_odd_all 
                                if covgal_per_band is not None:

                                    cov_even_save_per_band = []
                                    cov_odd_save_per_band = []
                                    for ii in range(covgal_per_band[0].shape[0]):
                                        cov_even_save_per_band.extend(covgal_per_band[0][ii][ii:])
                                    for ii in range(covgal_per_band[1].shape[0]):
                                        cov_odd_save_per_band.extend(covgal_per_band[1][ii][ii:])



                                count+=1





                    #res['mcal_g_1p'],res['mcal_g_1m'],res['mcal_g_2p'],res['mcal_g_2m'],res['mcal_g_noshear'],res['mcal_s2n_noshear']

                                # *********************
                                if (config['setup_image_sims']):
                                        if config['run_mcal']:
                                            try:
                                                tab_targets_m.g1_mcal.append(res_x['mcal_g_noshear'][0][0])
                                                tab_targets_m.g1p_mcal.append(res_x['mcal_g_1p'][0][0])
                                                tab_targets_m.g1m_mcal.append(res_x['mcal_g_1m'][0][0])
                                                tab_targets_m.g2_mcal.append(res_x['mcal_g_noshear'][0][1])
                                                tab_targets_m.g2p_mcal.append(res_x['mcal_g_2p'][0][1])
                                                tab_targets_m.g2m_mcal.append(res_x['mcal_g_2m'][0][1])
                                                tab_targets_m.mcal_sn.append(res_x['mcal_s2n_noshear'][0])
                                                tab_targets_m.mcal_sn_1p.append(res_x['mcal_s2n_1p'][0])
                                                tab_targets_m.mcal_sn_1m.append(res_x['mcal_s2n_1m'][0])
                                                tab_targets_m.mcal_sn_2p.append(res_x['mcal_s2n_2p'][0])
                                                tab_targets_m.mcal_sn_2m.append(res_x['mcal_s2n_2m'][0])
                                                tab_targets_m.mcal_flags.append(res_x['mcal_flags'][0])
                                                tab_targets_m.mcal_size_ratio.append(res_x['mcal_T_ratio_noshear'][0])
                                                tab_targets_m.mcal_size_ratio_1p.append(res_x['mcal_T_ratio_1p'][0])
                                                tab_targets_m.mcal_size_ratio_1m.append(res_x['mcal_T_ratio_1m'][0]) 
                                                tab_targets_m.mcal_size_ratio_2p.append(res_x['mcal_T_ratio_2p'][0])   
                                                tab_targets_m.mcal_size_ratio_2m.append(res_x['mcal_T_ratio_2m'][0])                        
                                            except:
                                                tab_targets_m.g1_mcal.append(-1)
                                                tab_targets_m.g1p_mcal.append(-1)
                                                tab_targets_m.g1m_mcal.append(-1)
                                                tab_targets_m.g2_mcal.append(-1)
                                                tab_targets_m.g2p_mcal.append(-1)
                                                tab_targets_m.g2m_mcal.append(-1)
                                                tab_targets_m.mcal_sn.append(-1)
                                                tab_targets_m.mcal_sn_1p.append(-1)
                                                tab_targets_m.mcal_sn_1m.append(-1)
                                                tab_targets_m.mcal_sn_2p.append(-1)
                                                tab_targets_m.mcal_sn_2m.append(-1)
                                                tab_targets_m.mcal_flags.append(-1)
                                                tab_targets_m.mcal_size_ratio.append(-1)
                                                tab_targets_m.mcal_size_ratio_1p.append(-1)
                                                tab_targets_m.mcal_size_ratio_1m.append(-1)
                                                tab_targets_m.mcal_size_ratio_2p.append(-1)
                                                tab_targets_m.mcal_size_ratio_2m.append(-1)


                                    # we need to print, index, Mf, stamp size
                                       # print (index)
                                       # print (tab_detections.images[index].imlist[0][0].shape)
                                       # mom = tab_detections.images[index].moments.get_moment(0.,0.)

                                       # print (np.sqrt(tab_detections.images[index].moments.get_covariance()[0][mom.M0,mom.M0]))

                                        if config['external']:
                                            tab_targets_m.add(mom, xy=newcent,id=index_t,covgal=MomentCovariance(covgal[0],covgal[1]))
                                        else:





                                            tab_targets_m.add(mom, xy=newcent,id=tab_detections.images[index].image_ID[0],covgal=MomentCovariance(covgal[0],covgal[1]))
                                        tab_targets_m.p0.append(p0)
                                        tab_targets_m.p0_PSF.append(p0_PSF)

                                        if config['external']:
                                            tab_targets_m.ra.append(external['ra'][index_t])
                                            tab_targets_m.dec.append(external['dec'][index_t])
                                        else:
                                            tab_targets_m.ra.append(newcent[0])
                                            tab_targets_m.dec.append(newcent[1])

                                        meb_ = np.array([m_.even for m_ in meb])
                                        tab_targets_m.meb.append(meb_[0,:])
                                        try:
                                            tab_targets_m.true_fluxes.append(fluxes)
                                        except:
                                            pass
                                        tab_targets_m.cov_odd_per_band.append(cov_odd_save_per_band)
                                        tab_targets_m.cov_even_per_band.append(cov_even_save_per_band)


                                        Mf,Mr,M1,M2,_ = tab_detections.images[index].moments_PSF.get_moment(0.,0.).even


                                        tab_targets_m.area.append(0.)




                                        tab_targets_m.psf_Mf.append(Mf)
                                        tab_targets_m.psf_Mr.append(Mr)
                                        tab_targets_m.psf_M1.append(M1)
                                        tab_targets_m.psf_M2.append(M2)
                                        nn = np.array([tab_detections.images[index].noise_rms[index_band][0] for index_band in range(tab_detections.images[index].n_bands)])
                                        #tab_targets_m.band1.append(nn[0])
                                        #tab_targets_m.band2.append(nn[1])
                                        #tab_targets_m.band3.append(nn[2])
                                        try:
                                            tab_targets_m.des_id.append(params_image_sims[p0]['des_id'])
                                            tab_targets_m.photoz.append(params_image_sims[p0]['photoz'])
                                        except:
                                            pass
                                else:
                                        if config['external']:
                                            tab_targets.add(mom, xy=newcent,id=index_t,covgal=MomentCovariance(covgal[0],covgal[1]))
                                        else:

                                            tab_targets.add(mom, xy=newcent,id=tab_detections.images[index].image_ID[0],covgal=MomentCovariance(covgal[0],covgal[1]))
                                        tab_targets.p0.append(p0)
                                        tab_targets.p0_PSF.append(p0_PSF)

                                        if config['external']:
                                            tab_targets.ra.append(external['ra'][index_t])
                                            tab_targets.dec.append(external['dec'][index_t])
                                        else:
                                            tab_targets.ra.append(newcent[0])
                                            tab_targets.dec.append(newcent[1])

                                        meb_ = np.array([m_.even for m_ in meb])
                                        tab_targets.meb.append(meb_[0,:])
                                        try:
                                            tab_targets.true_fluxes.append(fluxes)
                                        except:
                                            pass
                                        Mf,Mr,M1,M2,_ = tab_detections.images[index].moments_PSF.get_moment(0.,0.).even


                                        tab_targets.area.append(0.)    

                                        tab_targets.psf_Mf.append(Mf)
                                        tab_targets.psf_Mr.append(Mr)
                                        tab_targets.psf_M1.append(M1)
                                        tab_targets.psf_M2.append(M2)

                                        tab_targets.cov_odd_per_band.append(cov_odd_save_per_band)
                                        tab_targets.cov_even_per_band.append(cov_even_save_per_band)

                                        nn = np.array([tab_detections.images[index].noise_rms[index_band][0] for index_band in range(tab_detections.images[index].n_bands)])

                                        #tab_targets.band1.append(nn[0])
                                        #tab_targets.band2.append(nn[1])
                                        #tab_targets.band3.append(nn[2])   

                                        #try:
                                        if (config['setup_image_sims']):
                                            try:
                                                tab_targets.g1_mcal.append(res_x['mcal_g_noshear'][0][0])
                                                tab_targets.g1p_mcal.append(res_x['mcal_g_1p'][0][0])
                                                tab_targets.g1m_mcal.append(res_x['mcal_g_1m'][0][0])
                                                tab_targets.g2_mcal.append(res_x['mcal_g_noshear'][0][1])
                                                tab_targets.g2p_mcal.append(res_x['mcal_g_2p'][0][1])
                                                tab_targets.g2m_mcal.append(res_x['mcal_g_2m'][0][1])
                                                tab_targets.mcal_sn.append(res_x['mcal_s2n_noshear'][0])
                                                tab_targets.mcal_sn_1p.append(res_x['mcal_s2n_1p'][0])
                                                tab_targets.mcal_sn_1m.append(res_x['mcal_s2n_1m'][0])
                                                tab_targets.mcal_sn_2p.append(res_x['mcal_s2n_2p'][0])
                                                tab_targets.mcal_sn_2m.append(res_x['mcal_s2n_2m'][0])
                                                tab_targets.mcal_flags.append(res_x['mcal_flags'][0])
                                                tab_targets.mcal_size_ratio.append(res_x['mcal_T_ratio_noshear'][0])
                                                tab_targets.mcal_size_ratio_1p.append(res_x['mcal_T_ratio_1p'][0])
                                                tab_targets.mcal_size_ratio_1m.append(res_x['mcal_T_ratio_1m'][0]) 
                                                tab_targets.mcal_size_ratio_2p.append(res_x['mcal_T_ratio_2p'][0])   
                                                tab_targets.mcal_size_ratio_2m.append(res_x['mcal_T_ratio_2m'][0])      

                                            except:
                                                tab_targets.g1_mcal.append(-1)
                                                tab_targets.g1p_mcal.append(-1)
                                                tab_targets.g1m_mcal.append(-1)
                                                tab_targets.g2_mcal.append(-1)
                                                tab_targets.g2p_mcal.append(-1)
                                                tab_targets.g2m_mcal.append(-1)
                                                tab_targets.mcal_sn.append(-1)
                                                tab_targets.mcal_sn_1p.append(-1)
                                                tab_targets.mcal_sn_1m.append(-1)
                                                tab_targets.mcal_sn_2p.append(-1)
                                                tab_targets.mcal_sn_2m.append(-1)
                                                tab_targets.mcal_flags.append(-1)
                                                tab_targets.mcal_size_ratio.append(-1)
                                                tab_targets.mcal_size_ratio_1p.append(-1)
                                                tab_targets.mcal_size_ratio_1m.append(-1)
                                                tab_targets.mcal_size_ratio_2p.append(-1)
                                                tab_targets.mcal_size_ratio_2m.append(-1)


                                        #except:
                                        #    pass
                                        try:
                                            tab_targets.des_id.append(params_image_sims[p0]['des_id'])
                                            tab_targets.photoz.append(params_image_sims[p0]['photoz'])
                                        except:
                                            pass



                                if index%10:
                                    if not (config['setup_image_sims']):
                                        if len(bands_not_masked) >= config['minimum_number_of_bands']:
                                            #print ('adding detection stamp')
                                            
                                            # add detection stamp **************************************+++++
                                            tab_detections.images[index].Load_MEDS(index, meds = m_array)
                                            tab_detections.images[index].make_WCS()
                                            tab_detections.images[index].make_false_stamp(use_COADD_only=config['COADD_only'])
                                            tab_detections.images[index].zero_padd_psf()


                                            mute_range = [index,index+1]


                                            tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = config['COADD_only'], flags = 0, MOF_subtraction = False, band_dict = params_template['band_dict'], chunk_range = mute_range,pad_factor=config['pad_factor'])

                                            tab_targets.add(mom, xy=newcent,id=tab_detections.images[index].image_ID[0],covgal=MomentCovariance(covgal[0],covgal[1]))
                                            tab_targets.p0.append(p0)
                                            tab_targets.p0_PSF.append(p0_PSF)


                                            tab_targets.ra.append(newcent[0])
                                            tab_targets.dec.append(newcent[1])

                                            meb_ = np.array([m_.even for m_ in meb])
                                            tab_targets.meb.append(meb_[0,:])

                                            try:
                                                tab_targets.true_fluxes.append(fluxes)
                                            except:
                                                pass
                                            Mf,Mr,M1,M2,_ = tab_detections.images[index].moments_PSF.get_moment(0.,0.).even


                                            tab_targets.area.append(10.)    

                                            tab_targets.psf_Mf.append(Mf)
                                            tab_targets.psf_Mr.append(Mr)
                                            tab_targets.psf_M1.append(M1)
                                            tab_targets.psf_M2.append(M2)

                                            tab_targets.cov_odd_per_band.append(cov_odd_save_per_band)
                                            tab_targets.cov_even_per_band.append(cov_even_save_per_band)

                                            try:
                                                tab_targets.des_id.append(params_image_sims[p0]['des_id'])
                                                tab_targets.photoz.append(params_image_sims[p0]['photoz'])
                                            except:
                                                pass

                                            '''
                                            for index_band in range(3):
                                                start = 1
                                                end = tab_detections.images[index].ncutout[index_band]
                                                image_storage[index][index_band] = dict()
                                                for exp in range(start, end):  


                                                    image_storage[index][index_band][exp] = [tab_detections.images[index].imlist[index_band][exp],0,False]
                                            '''

                                if not (config['setup_image_sims']):
                                    if  config['MOF_subtraction'] :


                                        del tab_detections.images[index]
                                        Wide_g = Image(index, meds = m_array, bands = config['bands'])
                                        tab_detections.insert_image(Wide_g,index)
                                        
                                        tab_detections.images[index].Load_MEDS_fast(index, meds = m_array)
                                        tab_detections.images[index].MOF_models =  np.load(config['output_folder']+'/MOF_models/{0}_{1}.npy'.format(tile,index),allow_pickle=True)
                                        if index % 5:
                                            gc.collect()

                                    else:

                                        tab_detections.images[index].moments = None
                                        del tab_detections.images[index].moments  

                                        if not config['MOF_subtraction'] :
                                            tab_detections.images[index].wtlist = None
                                            tab_detections.images[index].jaclist = None
                                            tab_detections.images[index].MOF_model_rendered = None
                                            tab_detections.images[index].MOF_models = None

                                            tab_detections.images[index].rowcol = None
                                            tab_detections.images[index].orig_rowcol = None


                                            del tab_detections.images[index].wtlist 
                                            del tab_detections.images[index].jaclist 

                                            del tab_detections.images[index].rowcol 
                                            del tab_detections.images[index].orig_rowcol 



                                        tab_detections.images[index].imlist = None
                                        tab_detections.images[index].seglist = None
                                        tab_detections.images[index].masklist = None
                                        tab_detections.images[index].psf = None
                                        tab_detections.images[index].size = None
                                        tab_detections.images[index].mfrac_flag = None
                                        tab_detections.images[index].noise_rm = None




                                        del tab_detections.images[index].imlist 
                                        del tab_detections.images[index].seglist 
                                        del tab_detections.images[index].masklist 
                                        del tab_detections.images[index].psf 
                                        del tab_detections.images[index].size 
                                        del tab_detections.images[index].mfrac_flag 
                                        del tab_detections.images[index].noise_rm 
                                        del tab_detections.images[index].MOF_model_rendered


                                        if not config['MOF_subtraction'] :
                                            tab_detections.images[index] = None
                                            #del tab_detections.images[index]
                                        gc.collect()

                    except:
                        print ('index failed :', index,tile,chunk_range)
                print ('\n----\n')
        
               #for index_t in frogress.bar(range(chunk_range[0],chunk_range[1])):
               #    tab_detections.images[index].imlist = None
               #    tab_detections.images[index].seglist = None
               #    tab_detections.images[index].masklist = None
               #    tab_detections.images[index].wtlist = None
               #    tab_detections.images[index].jaclist = None
               #    tab_detections.images[index].psf = None
               #    tab_detections.images[index].MOF_model_rendered = None
               #    
               #    #del tab_detections.images[index].imlist 
               #    #del tab_detections.images[index].seglist
               #    #del tab_detections.images[index].masklist
               #    #del tab_detections.images[index].wtlist
               #    #del tab_detections.images[index].jaclist
               #    #del tab_detections.images[index].psf 
               #    #del tab_detections.images[index].MOF_model_rendered 
               #    
                #del tab_detections.images
                #gc.collect()
                
                        
                #print ('final moments ',np.array(tab_targets_m.moment)[:,0]/np.array(tab_targets.moment)[:,0])
               # prihdu,tbhdu = save(tab_targets)
               # for i in range(len(config['band_dict'])):
               #     prihdu.header['band_'+str(config['band_dict'][i][0])+'_weight'] =  config['band_dict'][i][1]
               # list_final = [prihdu]
               # list_final.append(tbhdu)
               # thdulist = fits.HDUList(list_final)
                
                #try:
                if config['setup_image_sims']:
                        save_(tab_targets,path+'_chunk_{0}.fits'.format(iii),config)
                        save_(tab_targets_m,(path+'_chunk_{0}.fits'.format(iii)).replace('ISp','ISm'),config)
                else:
                        save_(tab_targets,path+'_chunk_{0}.fits'.format(iii),config)
                       # save_obj(path+'_image_storage_chunk_{0}.fits'.format(iii),image_storage)
                        
                        
                      
                        
                del tab_detections
                tab_targets = None
                del tab_targets
                if config['setup_image_sims']:
                    tab_targets_m = None
                    del tab_targets_m
                gc.collect()

               # except:
               #     pass

