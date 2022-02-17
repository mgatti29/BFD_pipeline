import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
from .read_meds_utils import Image, MOF_table, DetectionsTable
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

version_split_string = (version.version).split('.')
astropy_version = np.float(version_split_string[0] + '.' + version_split_string[1])



def save(self):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        col.append(fits.Column(name="moments",format="5E",array=self.moment))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))
        col.append(fits.Column(name="number",format="K",array=self.id))
        
        col.append(fits.Column(name="ra",format="D",array=self.ra))
        col.append(fits.Column(name="dec",format="D",array=self.dec))
        
        PSF_moments = np.vstack([np.array(self.psf_Mf),np.array(self.psf_Mr),np.array(self.psf_M1),np.array(self.psf_M2)]).T
        col.append(fits.Column(name="psf_moments",format="4E",array=PSF_moments))

        col.append(fits.Column(name="w_i",format="D",array=self.band1))
        col.append(fits.Column(name="w_r",format="D",array=self.band2))
        col.append(fits.Column(name="w_z",format="D",array=self.band3))
        
        #col.append(fits.Column(name="NOISETIER",format="K",array=noisetier*np.ones_like(self.id)))
        
        if len(self.num_exp) == len(self.id):
            col.append(fits.Column(name="num_exp",format="K",array=self.num_exp))
        if self.cov_even is not None:
            # saved in order M0xM0, M0xMR, M0xM1, M0xM2, M0xMC,
            # MRxMR, MRxM1, MRxM2, MRxMC, M1xM1, M1xM2, M1xMC,
            # M2xM2, M2xMC, MCxMC (total of 15)
            col.append(fits.Column(name="cov_even",format="15E",array=self.cov_even))
            # saved in order MXxMX, MXxMY, MYxMY (total of 3)
            col.append(fits.Column(name="cov_odd",format="3E",array=self.cov_odd))
        if len(self.delta_flux_moment) == len(self.id):
            col.append(fits.Column(name="delta_flux_moment",format="E",array=self.delta_flux_moment))
        if len(self.cov_delta_flux_moment) == len(self.id):
            col.append(fits.Column(name="cov_delta_flux_moment",format="E",array=self.cov_delta_flux_moment))
            

        cols=fits.ColDefs(col)

        tbhdu = fits.BinTableHDU.from_columns(cols)
        self.prihdu.header['NLOST'] = self.nlost  # Update value
        tbhdu.header['WT_N'] = self.prihdu.header['WT_N'] 
        tbhdu.header['WT_SIG'] = self.prihdu.header['WT_SIG']

        #tbhdu.header['TIERNAME'] = noisetier
        return self.prihdu,tbhdu
    
def measure_moments_targets(output_folder,**config):
    '''
    It computes the moments from des y6 tiles.
    '''
    
    # Read the config file
    print ('Executing the measure_target_moments stage')
    
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
    for  tile in tiles_to_be_used:
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
        #path = output_folder+'/targets/targets_'+tile
        #if not os.path.exists(path+'_chunk_last'+'.pkl'):
        list_run.append(count)
    print ('Runs to do: ',len(list_run))
    config['output_folder'] = output_folder
    run_count = 0

    
    # Runs the main pipeline **********************************************************
    if config['MPI_per_tile']:
        while run_count<len(list_run):
            pipeline(config, dictionary_runs, list_run[run_count])
            run_count+=2
    else:
        while run_count<len(list_run):
            comm = MPI.COMM_WORLD
            #try:
            if (run_count+comm.rank) < len(list_run):
                pipeline(config, dictionary_runs, list_run[run_count+comm.rank])
    
            #except:
            #    pass
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
        
            
def pipeline(config, dictionary_runs, count):
        
        tile = list(dictionary_runs.keys())[count]
        print ('TILE running: ',tile)
        if config['setup_image_sims']:
            path = config['output_folder']+'/targets/ISp_targets_'+tile
        else:
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
        
        # if it's an image sims run, load the MOF parameters for PSF and galaxies.
        if config['setup_image_sims']:
            files = glob.glob(output_folder+'/gal_psf_params/chunk*')
            params_image_sims = dict()
            for file in files:
                mute_p = np.load(file,allow_pickle='TRUE').item()
                for key in mute_p.keys():
                    params_image_sims[key] = mute_p[key]
        
        # division in chunks *********
        
        len_file = np.min([len(tab_detections.images),config['max_target_per_tile']])
        chunk_size = config['chunk_size']
        
        pool = multiprocessing.Pool(processes=config['agents_chunk'])

        runs = math.ceil(len_file/chunk_size)
        xlist = range(runs)
        print ('subruns for this tile: ',runs,' chunk size: ', chunk_size)
        
        
        if config['MPI_per_tile']:
            run_count = 0
            comm = MPI.COMM_WORLD
            while run_count<runs:
                if run_count+comm.rank<runs:
                    f(run_count+comm.rank, config = config, params_template = params_template,chunk_size=chunk_size, path = path, tab_detections = tab_detections, m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs,params_image_sims = params_image_sims)
            
                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        else:
            _ = pool.map(partial(f, config = config, params_template = params_template,chunk_size=chunk_size, path = path, tab_detections = tab_detections, m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs,params_image_sims = params_image_sims), xlist)
            #for x in xlist:
            #    f(x, config = config,params_template = params_template, chunk_size=chunk_size, path = path, tab_detections = tab_detections, m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs)


def f(iii, config, params_template, chunk_size, path, tab_detections, m_array, bands, len_file, runs, params_image_sims):
            
            '''
            
            
            
            '''
            
            
            
        
            # inititalise target table
            
            tab_targets = TargetTable(n = params_template['n'],
                                  sigma = params_template['sigma'],
                                  cov=None)

            tab_targets.ra = []
            tab_targets.dec = []
            tab_targets.psf_Mf = []
            tab_targets.psf_Mr = []
            tab_targets.psf_M1= []
            tab_targets.psf_M2= []
            tab_targets.band1 = []
            tab_targets.band2 = []
            tab_targets.band3 = []
            
            if config['setup_image_sims']:
                tab_targets_m = copy.copy(tab_targets)
            
            m_array = [meds.MEDS(m_array[band]['meds']) for band in bands]
            chunk_range =  [int(chunk_size*iii),int(np.min([chunk_size*(iii+1),len_file]))] 
            start = timeit.default_timer()
            if iii == (runs)-1:
                path_save1 = path+'_chunk_last'.format(iii)
            else:
                path_save1 = path+'_chunk_{0}'.format(iii)

            if not os.path.exists(path_save1+'.fits'):
                for index in range(chunk_range[0],chunk_range[1]):
                    update_progress((index*1.+1-chunk_range[0])/(chunk_range[1]-chunk_range[0]),timeit.default_timer(),start)
                    # load meds for image 'index'
                    tab_detections.images[index].Load_MEDS(index, meds = m_array)
                    # read WCS 
                    tab_detections.images[index].make_WCS()
                    #setup flag
                    tab_detections.images[index].flags = 0
                    # check masked bands and only consider objects with 2+ bands not masked
                    bands_not_masked = tab_detections.images[index].check_mfrac(limit=config['frac_limit'], use_COADD_only = config['COADD_only'])
                    
                    
                    number_of_replicas = 1
                    if config['setup_image_sims']:
                        '''
                        Here I should create the image_ from image sims.
                        
                        '''
                        # the 2* is to allow positively and negatively sheared simulated images.
                        number_of_replicas = 2*config['number_of_replicas']
                        
                    # the following loop allows to use simulate images instead real galaxyes
                    for replica in range(number_of_replicas):
                        if config['setup_image_sims']:
                            
                            if replica%2 == 0 :
                                # generate a simulated image
                                sim_p,sim_m,sim_PSF = tab_detections.generate_simulated_images(index,config,params_image_sims,use_COADD_only=True)
                                tab_detections.images[index].imlist = copy.deepcopy(sim_p)
                                tab_detections.images[index].psf = copy.deepcopy(sim_PSF)
                            if replica%2 == 1 :
                                tab_detections.images[index].imlist = copy.deepcopy(sim_m)


                    
                    
                    
                        if config['interp_masking']:
                            #Interpolates ther images over masked pixels
                            tab_detections.images[index].deal_with_bmask(use_COADD_only = config['COADD_only'])

                        # remove exposures that don't pass the mask fraction.

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

                        if len(bands_not_masked) < config['minimum_number_of_bands']:
                            #print ('too many pixels masked')
                            tab_detections.images[index].flags = 1
                            # the image is not usable, so discard the images

                        else:
                            tab_detections.images[index].zero_padd_psf()


                            tab_detections.render_MOF_models(index = index, render_self = False, render_others = True)
                            mute_range = [index,index+1]

                            tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = config['COADD_only'], flags = 0, MOF_subtraction = config['MOF_subtraction'], band_dict = params_template['band_dict'], chunk_range = mute_range)



                            newcent=np.array([tab_detections.images[index].image_ra,tab_detections.images[index].image_dec])[:,0]+tab_detections.images[index].xyshift/3600.0
                            
                            
                            if (config['setup_image_sims']) and (replica%2 == 1 ):
                                    tab_targets_m.add(tab_detections.images[index].moments.get_moment(0.,0.), xy=newcent,id=tab_detections.images[index].image_ID[0],number=1,covgal=tab_detections.images[index].moments.get_covariance())

                                    tab_targets_m.ra.append(newcent[0])
                                    tab_targets_m.dec.append(newcent[1])

                                    Mf,Mr,M1,M2,_ = tab_detections.images[index].moments_PSF.get_moment(0.,0.).even


                                    tab_targets_m.psf_Mf.append(Mf)
                                    tab_targets_m.psf_Mr.append(Mr)
                                    tab_targets_m.psf_M1.append(M1)
                                    tab_targets_m.psf_M2.append(M2)
                                    nn = np.array([tab_detections.images[index].noise_rms[index_band][0] for index_band in range(tab_detections.images[index].n_bands)])
                                    tab_targets_m.band1.append(nn[0])
                                    tab_targets_m.band2.append(nn[1])
                                    tab_targets_m.band3.append(nn[2])

                            else:
                                    tab_targets.add(tab_detections.images[index].moments.get_moment(0.,0.), xy=newcent,id=tab_detections.images[index].image_ID[0],number=1,covgal=tab_detections.images[index].moments.get_covariance())

                                    tab_targets.ra.append(newcent[0])
                                    tab_targets.dec.append(newcent[1])

                                    Mf,Mr,M1,M2,_ = tab_detections.images[index].moments_PSF.get_moment(0.,0.).even

                                    tab_targets.psf_Mf.append(Mf)
                                    tab_targets.psf_Mr.append(Mr)
                                    tab_targets.psf_M1.append(M1)
                                    tab_targets.psf_M2.append(M2)
                                    nn = np.array([tab_detections.images[index].noise_rms[index_band][0] for index_band in range(tab_detections.images[index].n_bands)])
                                    tab_targets.band1.append(nn[0])
                                    tab_targets.band2.append(nn[1])
                                    tab_targets.band3.append(nn[2])   
                                    
                                    
                    tab_detections.images[index].imlist = None
                    tab_detections.images[index].seglist = None
                    tab_detections.images[index].masklist = None
                    tab_detections.images[index].wtlist = None
                    tab_detections.images[index].jaclist = None
                    tab_detections.images[index].psf = None
                    tab_detections.images[index].MOF_model_rendered = None
                    gc.collect()
                        
                
                    

                prihdu,tbhdu = save(tab_targets)
                for i in range(len(config['band_dict'])):
                    prihdu.header['band_'+str(config['band_dict'][i][0])+'_weight'] =  config['band_dict'][i][1]
                list_final = [prihdu]
                list_final.append(tbhdu)
                thdulist = fits.HDUList(list_final)
                # sometimes when doing MPI the code protests **
                if iii == (runs)-1:
                    fitsname = path+'_chunk_last.fits'.format(iii)
                else:
                    fitsname = path+'_chunk_{0}.fits'.format(iii)
                try:
                    if astropy_version >= 1.3:
                        thdulist.writeto(fitsname,overwrite=True)
                    else:
                        thdulist.writeto(fitsname,clobber=True)
                except:
                    pass
                
                if config['setup_image_sims']:
                    # also saves the m component
                    prihdu,tbhdu = save(tab_targets_m)
                    for i in range(len(config['band_dict'])):
                        prihdu.header['band_'+str(config['band_dict'][i][0])+'_weight'] =  config['band_dict'][i][1]
                    list_final = [prihdu]
                    list_final.append(tbhdu)
                    thdulist = fits.HDUList(list_final)
                    # sometimes when doing MPI the code protests **
                    if iii == (runs)-1:
                        fitsname = (path+'_chunk_last.fits'.format(iii)).replace('ISp','ISm')
                    else:
                        fitsname =(path+'_chunk_{0}.fits'.format(iii)).replace('ISp','ISm')
                    try:
                        if astropy_version >= 1.3:
                            thdulist.writeto(fitsname,overwrite=True)
                        else:
                            thdulist.writeto(fitsname,clobber=True)
                    except:
                        pass
    
                
