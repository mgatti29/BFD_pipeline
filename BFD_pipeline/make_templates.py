Copy code
import astropy.io.fits as fits
import astropy.version
import bfd
import bfd.momentcalc as mc
from bfd.momenttable import TemplateTable
from bfd import TierCollection, Moment
import copy
import frogress
import gc
import glob
import h5py as h5
import math
import meds
import multiprocessing
import ngmix.gmix as gmix
import numpy as np
import pandas as pd
import psfex
import scipy
import sys
import time
import timeit
from functools import partial

from .read_meds_utils import Image, MOF_table, DetectionsTable, save_template
from .utilities import save_obj, load_obj, select_obj_w, select_obj_w_special, bulkUnpack, rotate, angle

import argparse
import copy
import os
import pickle
import timeit
import glob
import numpy as np
import pandas as pd

def read_files_quick(ii,deep_files):
   """
    Extracts specific information from the given file and returns the extracted data in arrays.

    Parameters:
    - ii (int): Index of the file to process.
    - deep_files (list): List of file names.

    Returns:
    - ra (list): List of RA values.
    - dec (list): List of Dec values.
    - ra_DF (list): List of RA values (duplicate).
    - dec_DF (list): List of Dec values (duplicate).
    - MAG_I (list): List of I-band magnitude values.
    - tilename (list): List of tilename values.
    - p0_ (list): List of p0 values.
    - p0_PSF_ (list): List of p0_PSF values.
    - moments (numpy.ndarray): Array of even moments.
    - moments_odd (numpy.ndarray): Array of odd moments.
    - error (numpy.ndarray): Array of error values.
    - indexu (numpy.ndarray): Array of indexu values.
    - indexu_gal (numpy.ndarray): Array of indexu_gal values.
    - xyshift_detectinator (numpy.ndarray): Array of xyshift_detectinator values.
    - nblends (numpy.ndarray): Array of nblends values.
    - dmdg (numpy.ndarray): Array of dmdg values.
    - dmdgdg (numpy.ndarray): Array of dmdgdg values.
    """

    ra = []
    dec = []
    ra_DF = []
    dec_DF = []
    MAG_I = []
    tilename = []
    p0_ = []
    p0_PSF_ = []
    
    moments = [np.zeros(5)]
    moments_odd = [np.zeros(2)]
    error = [np.zeros(5)]
    indexu = [0]
    indexu_gal = [0]

    mf_per_band = []
    dmdg = [0]
    dmdgdg = [0]
        
    
    nblends = [0]
    xyshift_detectinator = [0]
    df = deep_files[ii]
    try:
        save_ = load_obj(df.strip('.pkl'))
        for index in (save_.keys()):

            mom = save_[index]['moments'].get_moment(0., 0.)

            moments.append(mom.even)
            moments_odd.append(mom.odd)
            error.append(np.sqrt(save_[index]['moments'].get_covariance()[0].diagonal()))
            indexu.append(save_[index]['index'])
            
            try:
                nblends.append(save_[index]['nblends'])
                xyshift_detectinator.append(save_[index]['xyshift_detectinator'])        
            except:
                nblends.append(1)
                xyshift_detectinator.append(1)
    
            try:
                dmdg.append(save_[index]['dMdg1'])        
                dmdgdg.append(save_[index]['dMdg1dg2'])            
            except:
                dmdg.append(0)
                dmdgdg.append(0)
   
            try:
                indexu_gal.append(save_[index]['index_gal'])
            except:
                indexu_gal.append(save_[index]['index'])
            
            try:
                mf_per_band.append(save_[index]['mf_per_band'])
            except:
                pass
            
            try:
                mof_index.append(save_[index]['MOF_index'])
                ra.append(save_[index]['ra'])
                dec.append(save_[index]['dec'])
                ra_DF.append(save_[index]['ra'])
                dec_DF.append(save_[index]['dec'])
                MAG_I.append(save_[index]['MAG_I'])
                tilename.append(save_[index]['tilename'])
            except:
                pass
            
            try:
                p0_.append(save_[index]['p0'])
                p0_PSF_.append(save_[index]['p0_PSF'])
            except:
                pass
    except:
        pass
    
    return [
        ra, dec, ra_DF, dec_DF, MAG_I, tilename, p0_, p0_PSF_,
        np.array(moments), np.array(moments_odd), np.array(error),
        np.array(indexu), np.array(indexu_gal), np.array(xyshift_detectinator),
        np.array(nblends), np.array(dmdg), np.array(dmdgdg)
    ]     




def pipeline(output_folder, config, deep_files, targets_properties, runs, index):
    """
    Runs a pipeline to generate translated version of the templates based on the provided inputs.

    Parameters:
    - output_folder (str): Path to the output folder.
    - config (dict): Configuration parameters.
    - deep_files (list): List of deep files.
    - targets_properties (dict): Dictionary of target properties.
    - runs (list): List of runs.
    - index (int): Index to select the run.

    Returns: None
    """

    t_index, d_index = runs[index]
    
    # Check if the templates file doesn't already exist
    if not os.path.exists(output_folder + '/templates/templates_junk/templates_{0}_{1}.pkl'.format(t_index, d_index)):
        try:
            templates_overview = fits.open(output_folder + '/templates_overview.fits')
            mfverview = templates_overview[1].data['moments'][:, 0]
            idverview = templates_overview[1].data['index_gal']
            class_w = templates_overview[1].data['class']
            wverview = templates_overview[1].data['w']
            overview_data = True
        except:
            overview_data = False

        # Set the sigma_xy and sigma_flux values from targets_properties
        config['sigma_xy'] = targets_properties[t_index]['sigma_xy']
        config['sigma_flux'] = targets_properties[t_index]['sigma_flux']

        # Create a TemplateTable instance with the specified configuration parameters
        tab_templates = TemplateTable(
            n=config['n'],
            sigma=config['sigma'],
            sn_min=config['sn_min'],
            sigma_xy=config['sigma_xy'],
            sigma_flux=config['sigma_flux'],
            sigma_step=config['sigma_step'],
            sigma_max=config['sigma_max'],
            xy_max=config['xy_max']
        )

        print('sigma_flux: ', config['sigma_flux'])
        print('sigma_XY: ', config['sigma_xy'])

        # Load the templates data from the appropriate file
        try:
            save_ = load_obj(output_folder + '/templates/' + 'templates_' + d_index)
        except:
            save_ = load_obj(output_folder + '/templates/' + 'IS_templates_' + d_index)

        cumulative_weight = 0
        cumulative_weight1 = 0
        templates = []
        count = 0
        count_g = 0
        print('number of entries: ', len(save_.keys()))

        # Downsample the templates based on the downsample_factor
        try:
            downsample_factor = config['downsample_factor']
            downsample = np.int(len(save_.keys()) * downsample_factor)
        except:
            downsample = len(save_.keys())
            downsample_factor = 1.

        # Iterate over the templates data and process each template
        for i_ in frogress.bar(range(np.int(len(save_.keys()) * downsample_factor))):
            i = list(save_.keys())[i_]

            start = timeit.default_timer()
            cc = False

            try:
                # Retrieve the moments for the template
                mom = save_[i]['moments'].get_moment(0., 0.)

                if overview_data:
                    # Retrieve additional data from the overview if available
                    idx = np.in1d(idverview, save_[i]['index'])
                    if wverview[idx] > 0:
                        w_ = wverview[idx][0]
                        cc = True

                        # Set even and odd moments from the overview data
                        mom.even = templates_overview[1].data['average_moments'][idx, :][0]
                        mom.odd = templates_overview[1].data['average_moments_odd'][idx, :][0]
                    else:
                        cc = False
                else:
                    # Set moments and weight for the template
                    mom = save_[i]['moments'].get_moment(0., 0.)
                    w_ = 1.
                    cc = True
            except:
                pass

            if cc:
                count_g += 1

                Mf = mom.even[mom.M0]
                SN_mute = Mf / np.sqrt(save_[i]['moments'].get_covariance()[0][mom.M0, mom.M0])
                mask_bool = True

                if mask_bool:
                    cumulative_weight1 += w_

                    # Generate templates from the moments
                    try:
                        flag_p, t, xy_kept, area_integral = save_[i]['moments'].make_templates(
                            config['sigma_xy'],
                            sigma_flux=config['sigma_flux'],
                            sn_min=config['sn_min'],
                            sigma_max=config['sigma_max'],
                            sigma_step=config['sigma_step'],
                            xy_max=config['xy_max']
                        )
                    except:
                        flag_p = False
                        t = []

                    ID_mute = copy.copy(save_[i]['index'])
                    ID_gal = copy.copy(save_[i]['index'])

                    if flag_p:
                        if overview_data:
                            class_ = class_w[idx]
                        else:
                            class_ = '-100_-100_-100'

                        gc.collect()
                        try:
                            p0 = save_[i]['p0']
                            p0_PSF = save_[i]['p0_PSF']
                        except:
                            p0 = 0
                            p0_PSF = 0

                        if len(t) > 0:
                            if t[0] is None:
                                continue
                            else:
                                for tmpl in t:
                                    count += 1
                                    tmpl.p0 = p0
                                    tmpl.p0_PSF = p0_PSF
                                    tmpl.id = ID_mute
                                    tmpl.id_gal = ID_gal
                                    tmpl.class_ = class_
                                    tmpl.area_integral = area_integral
                                    tmpl.nblends = save_[i]['nblends']
                                    tmpl.xyshift_detectinator = save_[i]['xyshift_detectinator']
                                    tmpl.xy_kept = xy_kept
                                    tmpl.nda *= w_ / downsample_factor
                                    templates.append(tmpl)
                                    cumulative_weight += tmpl.nda

        print('COUNT TEMPLATES ', count)

        # Save the generated templates to a file
        save_obj(output_folder + '/templates/templates_junk/templates_{0}_{1}'.format(t_index, d_index), templates)

        del save_
        gc.collect()

        
def make_templates(output_folder,**config):
    
    
    if config['MPI']:
        from mpi4py import MPI 
    print ('making templates')
    try:
        config['classes']
    except:
        print ('number of clases not provided. Setting to 7 (default)')
        config['classes'] = 7
    try:
        config['des_y3_match']
    except:
        print ('variable des_y3_match not provided. Setting to False (default)')
        config['des_y3_match'] = False
    try:
        config['reduce_input_files']
    except:
        print ('variable reduce_input_files not provided. Setting to False (default)')
        config['reduce_input_files'] = False
    try:
        config['min_sn_regrouping']
    except:
        print ('variable min_sn_regrouping not provided. Setting to 17 (default)')
        config['min_sn_regrouping'] = 17
    try:
        config['resume_templates_full']
    except:
        print ('variable resume_templates_full not provided. Setting to False (default)')
        config['resume_templates_full'] = False
        
    if ('compute' in config['stage']) or ('assembly' in config['stage']):
        
        # read list of targets :

        targets_properties = dict()
        
        # need to check if it exists. if it doesn't, exit.
        path_sims = output_folder+'/ISp_targets_sample_g.fits'
        path_data = output_folder+'/targets_sample_g.fits'
        if os.path.exists(path_sims):
            sims = True 
        elif os.path.exists(path_data):
            sims = False
        else:
            print ('no target file')
            sys.exit()
            
        
        if not sims:
            #targets = pf.open(output_folder+'/targets_sample_g.fits')
            deep_files = glob.glob(output_folder+'/templates/'+'/t*.pkl')

        else:
            #targets = pf.open(output_folder+'/ISp_targets_sample_g.fits')
            deep_files = glob.glob(output_folder+'/templates/'+'/IS*.pkl')

        # number of entries:

        #Let's open the noisetier file.
        NT_ = fits.open(output_folder+'/noisetiers.fits')
 
        noisetier = len(NT_)-1
        t_entries = np.arange(noisetier)

        for t_index,trt in enumerate(t_entries):
            targets_properties[trt] = dict()

            targets_properties[trt]['sigma_xy'] = np.float('{0:2.4f}'.format(np.sqrt(NT_[t_index+1].header['COVMXMX'])))
            targets_properties[trt]['sigma_flux'] = np.float('{0:2.4f}'.format((NT_[t_index+1].header['SIG_FLUX'])))
           # targets_properties[trt]['Mf_min'] =  np.float('{0:2.4f}'.format((NT_[t_index+1].header['FLUX_MIN']))) 


        #print (targets_properties)




        gc.collect()


        # eff area
        Adeep_files = glob.glob(output_folder+'/templates/'+'/A*.pkl')
        NN_templates = 0
        AREA = 0
        for d in Adeep_files:
            t_,a_,STAMP_SIM = load_obj(d.split('.pkl')[0])
            NN_templates += t_
            AREA += a_
        number_of_runs = noisetier*len(deep_files) #len(noise_tier)*len(deep_files)
        print ('number_of_runs: ', number_of_runs)
        print ('# deep fields: ',len(deep_files),'# noise tiers: ',noisetier)
        runs = []

        if not os.path.exists(output_folder+'/templates/templates_junk/'):
            try:
                os.mkdir(output_folder+'/templates/templates_junk/')
            except:
                pass
        for deep_file in deep_files:
            for i in range(noisetier):
                f = deep_file.split(output_folder+'/templates/')[1].split('templates_')[1].split('.pkl')[0]
                if not os.path.exists(output_folder+'/templates/templates_junk/templates_{0}_{1}.pkl'.format(i,f)):
                        runs.append([i,f])



        run_count = 0
        print ('runs to do [compute stage]: ',len(runs),' out of: ', len(deep_files)*(noisetier))
    if 'resume_templates' in config['stage']:   
       
        print ('resume templates')
        


        deep_files = glob.glob(output_folder+'/templates/'+'/IS_t*.pkl')
        if len(deep_files)==0:
            deep_files = glob.glob(output_folder+'/templates/'+'/t*.pkl')
            #if len(deep_files)==0:
            #    deep_files = glob.glob(output_folder+'/templates/'+'/old*.pkl')
                
           
        pool = multiprocessing.Pool(processes=6)
        xlist = np.arange(len(deep_files))
        uu = pool.map(partial(read_files_quick,deep_files=deep_files), xlist)


        ra = []
        dec = []
        ra_DF = []
        dec_DF = []
        MAG_I = []
        tilename = []
        p0_ = []
        p0_PSF_ = []
        moments = []
        moments_odd = []
        error = []
        indexu = []
        indexu_gal = []
        xyshift_detectinator = []
        nblends =[]
        dmdg = []
        dmdgdg = []

        for i in range(len(uu)):
            if i == 0:
                ra =      uu[i][0]
                dec =     uu[i][1]
                ra_DF =   uu[i][2]
                dec_DF =  uu[i][3]
                MAG_I =   uu[i][4]
                tilename =uu[i][5]
                p0_ =     uu[i][6]
                p0_PSF_ = uu[i][7]  
                moments = uu[i][8]  
                moments_odd = uu[i][9]  
                error =       uu[i][10]  
                indexu =      uu[i][11]  
                indexu_gal =  uu[i][12]  
                xyshift_detectinator =  uu[i][13]  
                nblends = uu[i][14]  
                dmdg =  uu[i][15]  
                dmdgdg =  uu[i][16]  
            else:
                ra = np.hstack([ra,uu[i][0]])
                dec =  np.hstack([dec,uu[i][1]])
                ra_DF =  np.hstack([ra_DF,uu[i][2]])
                dec_DF =  np.hstack([dec_DF,uu[i][3]])
                MAG_I =  np.hstack([MAG_I,uu[i][4]])
                tilename =  np.hstack([tilename,uu[i][5]])
                p0_ =  np.hstack([p0_,uu[i][6]])
                p0_PSF_ =  np.hstack([p0_PSF_,uu[i][7]])   
                moments     =  np.vstack([moments    ,uu[i][8]])
                moments_odd =  np.vstack([moments_odd,uu[i][9]])
                error       =  np.vstack([error      ,uu[i][10]])
                indexu      =  np.hstack([indexu     ,uu[i][11]])
                indexu_gal  =  np.hstack([indexu_gal ,uu[i][12]])
                xyshift_detectinator =  np.hstack([xyshift_detectinator,uu[i][13]  ])
                nblends = np.hstack([nblends,uu[i][14]  ])
                dmdg =  np.hstack([dmdg,uu[i][15]  ])
                dmdgdg =  np.hstack([dmdgdg,uu[i][16]  ])
                # print (ra.shape,xyshift_detectinator.shape,nblends.shape)
                
        # match with des y3 catalog ----
        print ('des_y3_match ', config['des_y3_match'])
        if config['des_y3_match']:
            dd = h5.File(config['des_y3_match'])
            dd = np.array(dd['data']['table'])
            mask_df = (dd['MASK_FLAGS']==0) & (dd['FLAGS']==0) & (dd['FLAGS_NIR']==0)& (dd['MASK_FLAGS_NIR']==0)  & (dd['KNN_CLASS']==1) #(m['KNN_CLASS']==1)])       
            #star_galaxy_sep = np.array(moments[:,1])/np.array(moments[:,0])<config['star_galaxy_sep']
            id_mask = np.in1d(np.array(indexu),dd['ID'][mask_df])
            
            moments = np.array(moments)[id_mask,:]
            moments_odd = np.array(moments_odd)[id_mask,:]
            error = np.array(error)[id_mask,:]
            try:
                mf_per_band = np.array(mf_per_band)[id_mask,:]
            except:
                pass
            
           
            indexu = np.array(indexu)[id_mask]
            indexu_gal = np.array(indexu_gal)[id_mask]
            
            xyshift_detectinator = np.array(xyshift_detectinator)[id_mask]
            nblends = np.array(nblends)[id_mask]
            
            dmdg  = np.array(dmdg)[id_mask]
            dmdgdg  = np.array(dmdgdg)[id_mask]
            try:
                mof_index = mof_index[id_mask]
                ra = ra[id_mask]
                dec = dec[id_mask]
                ra_DF = ra_DF[id_mask]
                dec_DF = dec_DF[id_mask]
                MAG_I = MAG_I[id_mask]
                tilename = tilename[id_mask]
            except:
                pass
            try:
                p0_ = p0_[id_mask]
                p0_PSF_ = p0_PSF_[id_mask]
            except:
                pass
        
        if config['resume_templates_full']:
            col=[]

            col.append(fits.Column(name="index",format="K",array=np.array(indexu)))
            col.append(fits.Column(name="index_gal",format="K",array=np.array(indexu_gal)))

            try:
                col.append(fits.Column(name="mf_per_band",format="{0}E".format(mf_per_band.shape[0]),array=np.array(mf_per_band)))
            except:
                pass
            col.append(fits.Column(name="moments",format="5E",array=np.array(moments)))
            col.append(fits.Column(name="moments_odd",format="5E",array=np.array(moments_odd)))


            col.append(fits.Column(name="error_moments",format="5E",array=np.array(error)))

            try:
                col.append(fits.Column(name="mof_index",format="K",array=np.array(mof_index)))
                col.append(fits.Column(name="ra",format="E",array=np.array(ra)))
                col.append(fits.Column(name="dec",format="E",array=np.array(dec)))
                col.append(fits.Column(name="ra_DF",format="E",array=np.array(ra_DF)))
                col.append(fits.Column(name="dec_DF",format="E",array=np.array(dec_DF)))
                col.append(fits.Column(name="MAG_I",format="E",array=np.array(MAG_I)))
                col.append(fits.Column(name="tilename",format="128A",array=np.array(tilename)))
            except:
                pass

            try:
                col.append(fits.Column(name="p0",format="E",array=np.array(p0_)))
                col.append(fits.Column(name="p0_PSF",format="E",array=np.array(p0_PSF_)))
            except:
                pass


            try:
                col.append(fits.Column(name="xyshift_detectinator",format="E",array=np.array(xyshift_detectinator)))
                col.append(fits.Column(name="nblends",format="E",array=np.array(nblends)))
            except:
                pass     

            tbhdu = fits.BinTableHDU.from_columns(col)
            prihdu = fits.PrimaryHDU()
            thdulist = fits.HDUList([prihdu,tbhdu])
            try:
                thdulist.writeto(output_folder+'/templates_overview_full.fits',overwrite=True)
            except:
                pass    

            

            
        
        moments = np.array(moments)

        # Rotate all the templates:
        phi = -0.5*(angle(moments[:,:].T))
        e_,o_ = rotate(moments[:,:].T,moments[:,:2].T,phi)
        new_DV = np.hstack([e_,o_])
        # load noisetiers ------
        nt = fits.open(output_folder+'/noisetiers.fits')

        #cholesky decomposition of the best tier
        u,o = bulkUnpack(nt[1].data['COVARIANCE'][:,:])
        u[-1],o[-1]
        new_cov = np.zeros((7,7))
        new_cov[:5,:5] = u[1]
        new_cov[5:,5:] = o[1]                  

        L = np.linalg.cholesky(np.linalg.inv(new_cov))

        # project the DV
        new_DV_ = np.matmul(new_DV,L)
        #
        mask = np.array([True,True,True,False,False,False,False])
        new_DV_ = new_DV_[:,mask]


        
        
        
        # let us apply this only to objects with low SN
        mask = new_DV[:,0]/np.sqrt(u[1][0,0])< config['min_sn_regrouping']
        
        
        #mask = (new_DV[:,0]/np.sqrt(u[-1][0,0])< 10000000.)
        print ('')
        print ('re-grouping {0} / {1}'.format(len(mask[mask]),len(mask)))
        print ('')
        
        mask_w,w,even_new,odd_new = select_obj_w(new_DV_[mask,:],np.array(moments)[mask,:],np.array(moments_odd)[mask,:],config['sigma_group_templates'])  
        
        
        newweigths = np.ones(len(mask))
        newweigths[mask] = w
        
        mask = (new_DV[:,0]/np.sqrt(u[-1][0,0])< 10000000.)
        
   

        # make classes ********************
    

        # Rotate all the templates:
        phi = -0.5*(angle(moments[:,:].T))
        e_,o_ = rotate(moments[:,:].T,moments[:,:2].T,phi)
        new_DV = np.hstack([e_,o_])
        # load noisetiers ------
        nt = fits.open(output_folder+'/noisetiers.fits')


        #cholesky decomposition of the worst tier
        u,o = bulkUnpack(nt[-1].data['COVARIANCE'][:,:])
        u[-1],o[-1]
        new_cov = np.zeros((7,7))
        new_cov[:5,:5] = u[1]
        new_cov[5:,5:] = o[1]                  

        L = np.linalg.cholesky(np.linalg.inv(new_cov))

        # project the DV
        new_DV_ = np.matmul(new_DV,L)
        #
        mask = np.array([True,True,True,False,False,False,False])
        new_DV_ = new_DV_[:,mask]

        # create classes for templates ************
        bins_ = np.exp(np.linspace(0,7,config['classes']))
        bins_[0] = -100
        class_t= np.digitize(new_DV_,bins_)
                    
        #class_t = (new_DV_/config['classes']).astype(np.int)
        class_ = np.array(['{0}_{1}_{2}'.format(x[0],x[1],x[2]) for x in class_t])

        #typical SN of the wide field galaxies of these templates ********************
        # a upper cut on sn must be based on the worst tier. a lowe cut on sn must be based on worst tier.

        # lower cut **
        u,o = bulkUnpack(nt[1].data['COVARIANCE'][:,:])
        mask_l = moments[:,0]/np.sqrt(u[1,0,0]) > config['sn_min'] 

        # upper cut **
        u,o = bulkUnpack(nt[-1].data['COVARIANCE'][:,:])
        mask_u = moments[:,0]/np.sqrt(u[1,0,0]) < config['sn_max']

        mask_total = mask_u & mask_l & (newweigths>0) & (xyshift_detectinator<5)
        
        class_[~mask_total] = '-100_-100_-100'

                    
            

        moments = np.array(moments)[mask_total,:]
        moments_odd = np.array(moments_odd)[mask_total,:]
        error = np.array(error)[mask_total,:]
        indexu = np.array(indexu)[mask_total]
        newweigths = np.array(newweigths)[mask_total]
        indexu_gal = np.array(indexu_gal)[mask_total]

        try:
            mf_per_band  = np.array(error)[mf_per_band,:]
        except:
            pass
        try:
            mof_index = mof_index[mask_total]
            ra = ra[mask_total]
            dec = dec[mask_total]
            ra_DF = ra_DF[mask_total]
            dec_DF = dec_DF[mask_total]
            MAG_I = MAG_I[mask_total]
            tilename = tilename[mask_total]
        except:
            pass
        try:
            p0_ = np.array(p0_)[mask_total]
            p0_PSF_ = np.array(p0_PSF_)[mask_total]
        except:
            pass
        
        try:
            
            xyshift_detectinator = xyshift_detectinator[mask_total]
            nblends = nblends[mask_total]
            dmdg = dmdg[mask_total]
            dmdgdg = dmdgdg[mask_total]
        except:
            pass
            
        class_ = class_[mask_total] 
        
        
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        
        col.append(fits.Column(name="index",format="K",array=np.array(indexu)))
        col.append(fits.Column(name="index_gal",format="K",array=np.array(indexu_gal)))
        
        
        try:
            
            col.append(fits.Column(name="mf_per_band",format="{0}E".format(mf_per_band.shape[0]),array=np.array(mf_per_band)))
        
        except:
            pass
        col.append(fits.Column(name="moments",format="5E",array=np.array(moments)))
        col.append(fits.Column(name="moments_odd",format="5E",array=np.array(moments_odd)))
        
        #col.append(fits.Column(name="average_moments",format="5E",array=np.array(even_new)))
        #col.append(fits.Column(name="average_moments_odd",format="5E",array=np.array(odd_new)))
          
        
        col.append(fits.Column(name="error_moments",format="5E",array=np.array(error)))
        col.append(fits.Column(name="w",format="E",array=np.array(newweigths)))
        
        
        try:
            col.append(fits.Column(name="mof_index",format="K",array=np.array(mof_index)))
            col.append(fits.Column(name="ra",format="E",array=np.array(ra)))
            col.append(fits.Column(name="dec",format="E",array=np.array(dec)))
            col.append(fits.Column(name="ra_DF",format="E",array=np.array(ra_DF)))
            col.append(fits.Column(name="dec_DF",format="E",array=np.array(dec_DF)))
            col.append(fits.Column(name="MAG_I",format="E",array=np.array(MAG_I)))
            col.append(fits.Column(name="tilename",format="128A",array=np.array(tilename)))
        except:
            pass
        
        try:
            col.append(fits.Column(name="p0",format="E",array=np.array(p0_)))
            col.append(fits.Column(name="p0_PSF",format="E",array=np.array(p0_PSF_)))
            
        except:
            pass
        

        try:
            col.append(fits.Column(name="xyshift_detectinator",format="E",array=np.array(xyshift_detectinator)))
            col.append(fits.Column(name="nblends",format="E",array=np.array(nblends)))
            col.append(fits.Column(name="dmdg",format="E",array=np.array(dmdg)))
            col.append(fits.Column(name="dmdgdg",format="E",array=np.array(dmdgdg)))
        except:
                pass    
            
        col.append(fits.Column(name="class",format="128A",array=np.array(class_)))
        tbhdu = fits.BinTableHDU.from_columns(col)
        prihdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([prihdu,tbhdu])
        try:
            thdulist.writeto(output_folder+'/templates_overview.fits',overwrite=True)
        except:
            pass
        
        
        if config['reduce_input_files']:
            try:
                templates_overview = fits.open(output_folder+'/templates_overview.fits')
                mfverview = templates_overview[1].data['moments'][:,0]
                idverview = templates_overview[1].data['index_gal']
                class_w = templates_overview[1].data['class']
                wverview = templates_overview[1].data['w']
                overview_data = True
            except:
                overview_data = False

            if overview_data:
                for file in frogress.bar(deep_files):
                    name = (file.split('.pkl')[0]).split(output_folder+'/templates/')[1]
                    path = output_folder+'/templates/'
                    try:
                        if not os.path.exists( output_folder+'/templates/old/'):
                            os.mkdir(output_folder+'/templates/old/')
                    except:
                        pass
                    try:
                    #if 1==1:
                       # if os.path.exists(path+'old_'+name+'.pkl'):
                       #     m = load_obj(path+'old_'+name)
                       #     old_exists = True
                       # else:
                       #     old_exists = False
                        m = load_obj(path+name)
                        if not os.path.exists(output_folder+'/templates/old/'+path+name+'.pkl'):
                             os.system('mv {0} {1}'.format(path+name+'.pkl',output_folder+'/templates/old/'))
                        #    if not old_exists:
                        #        os.system('mv {0} {1}'.format(path+name+'.pkl',path+'old_'+name+'.pkl'))
                        #if not old_exists:
                        #    os.system('mv {0} {1}'.format(path+name+'.pkl',path+'old_'+name+'.pkl'))
                        index_gal = []
                        for ii in ((m.keys())):
                            index_gal.append(m[ii]['index'])
                        index_gal = np.array(index_gal)


                        df = pd.DataFrame(data = {'w':wverview.byteswap().newbyteorder()}, index = idverview)
                        df1 = pd.DataFrame( index = index_gal)
                        u = df1.join(df,how = 'left',sort=False).dropna()
                        u = u[u['w']>0]
                        to_del = np.array(list(m.keys()))[~np.in1d(index_gal,np.array(u.index))]
                        for i in to_del:
                            del m[i]

                        tot_templ = len(m)
                        chunks = math.ceil(tot_templ/1000)
                        if tot_templ>0:
                            for ch in range(chunks):
                                m1 = copy.deepcopy(m)
                                for iii,kk in enumerate(list(m.keys())):
                                    if not ((iii>1000*ch) & (iii<=1000*(ch+1))):
                                        del m1[kk]
                                save_obj(output_folder+'/templates/'+(file.split('.pkl')[0]+'_subchunk_{0}'.format(ch)).split(output_folder+'/templates/')[1],m1)
                                del m1

                        del m
                        gc.collect()
    
                    except:
                        print ('failed ',path+name)  


    if 'compute' in config['stage']:
       
        while run_count<len(runs):
            comm = MPI.COMM_WORLD
            print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
            if (run_count+comm.rank) < len(runs):
                #try:
                    pipeline(output_folder,config, deep_files, targets_properties, runs, run_count+comm.rank)
               # except:
                #    pass
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
   
        print ('done compute')
    
    if 'assembly' in config['stage']:
        
        run_count = 0
        print ('noisetiers ',noisetier)
        while run_count <noisetier:

                comm = MPI.COMM_WORLD
                print(" I'm rank %d from %d running in total..." % (comm.rank, comm.size))

                if (run_count+comm.rank) < noisetier:
                    t_index = t_entries[run_count+comm.rank]
                    if not os.path.exists(output_folder+'/templates_NOISETIER_{0}.fits'.format(t_index)):
                       



                        config['sigma_xy'] =   targets_properties[t_index]['sigma_xy']
                        config['sigma_flux'] = targets_properties[t_index]['sigma_flux']



                        tab_templates = TemplateTable(n = config['n'],
                                    sigma = config['sigma'],
                                    sn_min = config['sn_min'], 
                                    sigma_xy = config['sigma_xy'], 
                                    sigma_flux = config['sigma_flux'], 
                                    sigma_step = config['sigma_step'], 
                                    sigma_max = config['sigma_max'],
                                    xy_max = config['xy_max'])


                        files = glob.glob(output_folder+'/templates/templates_junk/templates_{0}_*.pkl'.format(t_index))
                        count = 0
                        print ('number of templates files: ',len(files))
                        print (AREA*NN_templates)
                        print (AREA)
                        print (NN_templates)

                        if not STAMP_SIM:
                            EFFAREA = AREA
                        else:
                            EFFAREA = NN_templates

                        count = 0
                        for ii in frogress.bar(range(len(files))):
                            file = files[ii]
                            try:

                                template_list = load_obj(file.strip('.pkl'))

                                for jj_ in (range((len(template_list)))):

                                    tmpl = template_list[jj_]
                                    #tmpl.id = count
                                    #count += 1
                                    try:
                                        tmpl.nda  *= 1./EFFAREA*config['correction_factor_selection']
                                    except:

                                        tmpl.nda  *= 1./EFFAREA
                                    tab_templates.add(tmpl)
                                    count += 1
                            except:
                                print ('failed ','.'+file.strip('.pkl'))

                        end = timeit.default_timer()
                        print ('saving NOISETIER #',t_index, ', # templates: ',(len(tab_templates.templates))   )

                        save_template(tab_templates,output_folder+'/templates_NOISETIER_{0}.fits'.format(t_index),EFFAREA,t_index)
                        
                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        print ('done assembly')


        
    if 'group_templates' in config['stage']:
        run_count = 0

        NT_ = fits.open(output_folder+'/noisetiers.fits') 
        noisetier = len(NT_)-1
        while run_count <noisetier:

                comm = MPI.COMM_WORLD
                print(" I'm rank %d from %d running in total..." % (comm.rank, comm.size))

                if (run_count+comm.rank) < noisetier:
      
    
    
       
                    #output_folder = '/global/cfs/cdirs/des/mgatti/BFD_targets/BFD_targets_0/'
                    fitsname  = output_folder+'templates_NOISETIER_{0}_compact.fits'.format(run_count+comm.rank)
                    if not os.path.exists(fitsname):
                        print (fitsname)
                        file = output_folder+'templates_NOISETIER_{0}.fits'.format(run_count+comm.rank)

                        mute = fits.open(file)
                        moments = mute[1].data['moments']

                        md1 = mute[1].data['moments_dg1']
                        md2 = mute[1].data['moments_dg2']
                        mdm = mute[1].data['moments_dmu']
                        md1d1 = mute[1].data['moments_dg1_dg1']
                        md2d2 = mute[1].data['moments_dg2_dg2']
                        md1d2 = mute[1].data['moments_dg1_dg2']
                        mdmd1 = mute[1].data['moments_dmu_dg1']
                        mdmd2 = mute[1].data['moments_dmu_dg2']
                        mdmdm = mute[1].data['moments_dmu_dmu']
                        w = mute[1].data['weight']
                        j = mute[1].data['jSuppress']
                        info = np.hstack([moments,md1,md2,mdm,md1d1,md2d2,md1d2,mdmd1,mdmd2,mdmdm,j.reshape(-1,1)])

                        nt = fits.open(output_folder+'/noisetiers.fits')


                        # set concentration to 0 **********************************
                        moments[:,-1] = 0.

                        u,o = bulkUnpack(nt[run_count+comm.rank+1].data['COVARIANCE'][:,:])
                        u[-1],o[-1]
                        new_cov = np.zeros((7,7))
                        new_cov[:5,:5] = u[1]
                        new_cov[5:,5:] = o[1]     
                        L = np.linalg.cholesky(np.linalg.inv(new_cov))
                        new_DV_ = np.matmul(moments,L)

                        mask_w,w_new,new_info = select_obj_w_special(new_DV_,w,np.array(info),0.7)  

                  
                        
                        maskw = w_new!=0
                        ix = 0
                        col1 = fits.Column(name="id",format="K",array=np.arange(new_info.shape[0])[maskw])
                        ix = 0
                        col2 = fits.Column(name="moments",format="7E",array=new_info[maskw,ix:ix+7])
                        ix  += 7 
                        col3 = fits.Column(name="moments_dg1",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col4 = fits.Column(name="moments_dg2",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col5 = fits.Column(name="moments_dmu",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col6 = fits.Column(name="moments_dg1_dg1",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col8 = fits.Column(name="moments_dg2_dg2",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col7 = fits.Column(name="moments_dg1_dg2",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col9 = fits.Column(name="moments_dmu_dg1",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col10 = fits.Column(name="moments_dmu_dg2",format="7E",array=new_info[maskw,ix:ix+7])
                        ix += 7
                        col11= fits.Column(name="moments_dmu_dmu",format="7E",array=new_info[maskw,ix:ix+7])
                        col12= fits.Column(name="weight",format="E",array=w_new[maskw])
                        ix += 7
                        col13= fits.Column(name="jSuppress",format="E",array=new_info[maskw,ix:ix+1])


                        cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13])
                        tbhdu = fits.BinTableHDU.from_columns(cols, header=mute[1].header)
                        prihdu = fits.PrimaryHDU()
                        thdulist = fits.HDUList([prihdu,tbhdu])
                        thdulist.writeto(fitsname,overwrite=True)


                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
