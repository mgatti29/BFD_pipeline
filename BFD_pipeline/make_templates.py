import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable
from .read_meds_utils import Image, MOF_table, DetectionsTable, save_template
from .utilities import update_progress, save_obj, load_obj
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmix.gmix as gmix            
import timeit
import pickle
import time
import glob
import argparse
import os
import multiprocessing
from functools import partial
from astropy import version 
import gc
import astropy.io.fits as fits
import sys
import time
import timeit
import copy
import frogress
import scipy

import copy
import numpy as np
from bfd import TierCollection
import numpy as np
from bfd import Moment

def select_obj_w(new_DV,even,odd,radius):
    
    even_new = copy.deepcopy(even)
    odd_new =  copy.deepcopy(odd)
    
  
    mask = np.array([True]*new_DV.shape[0])
    ipx_ref = np.arange(new_DV.shape[0])
    
    w = np.ones((new_DV.shape[0]))
    r = len(mask[mask])
    redoit = True
    iter_ = 0
    # this removes objects too close ++++++++++++
    while redoit:
        iter_ += 1
        mask_w_dummy = copy.deepcopy(mask)
        
        YourTreeName = scipy.spatial.cKDTree(new_DV[mask_w_dummy], leafsize=100)
        
        d,ipx__ = YourTreeName.query(new_DV[mask_w_dummy], k=2, distance_upper_bound=radius)

        ipx_ = []
        for dd in ipx__:
            ipx_.append(np.sort(dd))
        ipx_ = np.array(ipx_)
        
        
        a = np.zeros((ipx_ref.shape[0]))

        u_ = (ipx_[ipx_[:,1]<len(mask[mask_w_dummy])])
        
        ipx_ref_ = ipx_ref[mask_w_dummy]
        for ii in frogress.bar(range(len(u_))):
            ipx = u_[ii]
            if (a[ipx_ref_[ipx[0]]] ==0) and (a[ipx_ref_[ipx[1]]] == 0): 
                mask[ipx_ref_[ipx[1]]] = False
                # remove obj list
                w[ipx[0]] += w[ipx[1]]
                
                even_new[ipx[0],:] += even[ipx[1],:]
                odd_new[ipx[0],:] += odd[ipx[1],:]      
                
                w[ipx[1]] = 0
                a[ipx_ref_[ipx[0]]] = 1
                a[ipx_ref_[ipx[1]]] = 1

        if r == len(mask[mask]):
            redoit = False
        r = len(mask[mask])
   

    for i in range(len(w)):
        even_new[i,:] /= w[i]
        odd_new[i,:] /= w[i]


    # sometimes, removed objects are too many. ++++++++++++
    #print (new_DV[mask,:])
    ipx_ref_ = ipx_ref[~mask_w_dummy]
    add_ = new_DV[~mask_w_dummy,:]
    removed_excess = 0
    base = len(new_DV[mask,0])
    for i in frogress.bar(range(add_.shape[0])):
        if len(new_DV[mask,0])!=base:
            base = len(new_DV[mask,0])
            YourTreeName = scipy.spatial.cKDTree(new_DV[mask], leafsize=100) 
        d,ipx_ = YourTreeName.query(add_[i].reshape(1,-1), k=1, distance_upper_bound=3*radius)

        #if i % 5000 == 0:
        #    print (' removed excess: ' ,removed_excess)
            
        if d>radius:
            #print ('put back: ',x[ipx_ref_[i]])
            #print (x[ipx_ref_[i]])
            #mask[ipx_ref_[i]] = True
            removed_excess +=1

    print ('\n removed excess: ' ,removed_excess)
    return mask,w,even_new,odd_new


def bulkUnpack(packed):
        '''Convert an Nx15 packed 1d version of even matrix into Nx5x5 array'''
        m = Moment()
        out = np.zeros( (packed.shape[0],m.NE,m.NE), dtype=float)
        
        j=0
        for i in range(m.NE):
            nvals = m.NE - i
            out[:,i,i:] = packed[:,j:j+nvals]
            out[:,i:,i] = packed[:,j:j+nvals]
            j += nvals
            
        odd = np.zeros( (packed.shape[0],m.NO, m.NO), dtype=float)
        odd[:,m.MX, m.MX] = 0.5*(out[:,m.M0,m.MR]+out[:,m.M0,m.M1])
        odd[:,m.MY, m.MY] = 0.5*(out[:,m.M0,m.MR]-out[:,m.M0,m.M1])
        odd[:,m.MX, m.MY] =  0.5*out[:,m.M0,m.M2]
        odd[:,m.MY, m.MX] =  0.5*out[:,m.M0,m.M2]
            
            
        return out,odd
    
def rotate(e,o,phi):
    '''Return Moment instance for this object rotated by angle phi radians
    '''
    e = copy.deepcopy(e)
    o = copy.deepcopy(o)
    #e = np.array(self.even)
    z = (e[2] + 1j*e[3]) * np.exp(2j*phi)
    e[2] = z.real
    e[3] = z.imag
    z = (o[0] + 1j*o[1]) * np.exp(1j*phi)
    o = np.array([z.real,z.imag])
    
    return e.T,o.T

def angle(e):
    return np.arctan((e[3]/e[2]))


def pipeline(output_folder,config, deep_files, targets_properties, runs, index):
   
        t_index, d_index = runs[index]

        try:
            templates_overview = pf.open(output_folder+'/templates_overview.fits')
            mfverview = templates_overview[1].data['moments'][:,0]
            idverview = templates_overview[1].data['index']
            wverview = templates_overview[1].data['w']
            overview_data = True
        except:
            overview_data = False
            pass
        config['sigma_xy'] =   targets_properties[t_index]['sigma_xy']
        config['sigma_flux'] = targets_properties[t_index]['sigma_flux']
        
        tab_templates = TemplateTable(n = config['n'],
                        sigma =  config['sigma'],
                        sn_min = config['sn_min'], 
                        sigma_xy = config['sigma_xy'], 
                        sigma_flux = config['sigma_flux'], 
                        sigma_step = config['sigma_step'], 
                        sigma_max = config['sigma_max'],
                        xy_max = config['xy_max'])


        limits = {'SN_Mf_min': config['sn_min'],
                  'SN_Mf_max': config['sn_max'],
                  'Mf_max': config['Mf_max'],
                  'Mf_min': config['Mf_min']}#targets_properties[t_index]['Mf_min']}


        print ('sigma_flux: ', config['sigma_flux'])
        print ('sigma_XY: ', config['sigma_xy'])
          
            
        #try:
        if 1==1:
            try:
                save_ = load_obj(output_folder+'/templates/'+'templates_'+d_index)
            except:
                save_ = load_obj(output_folder+'/templates/'+'IS_templates_'+d_index)
            
            templates = []
            count = 0
            count_g = 0
            print ('number of entries: ', len(save_.keys()))
            
            #for i in range(len(tab_mute.images)):
            # downsample
            
            try:
                
                downsample_factor = config['downsample_factor']
                downsample = np.int(len(save_.keys())*downsample_factor)
            except:
                downsample = len(save_.keys())
                downsample_factor=1.
            
    
            for i_ in frogress.bar(range( np.int(len(save_.keys())*downsample_factor))):
                    
                    i = list(save_.keys())[i_]

                    
                    
                    start = timeit.default_timer()
                    #update_progress(1.*i/(len(tab_mute.images)),timeit.default_timer(),start)           
                    cc = False
     
                    try:
                    #if 1==1:
                        #mom = tab_mute.images[i].moments.get_moment(0.,0.)
                        mom = save_[i]['moments'].get_moment(0.,0.)

                        if overview_data:
                            idx = np.in1d(mfverview,mom.even[mom.M0])
                            #print (wverview[idx],idverview[idx],i,save_[i]['index'])
                            if wverview[idx]>0:
                                w_ = wverview[idx]
                                cc =True
                                mom.even = templates_overview[1].data['average_moments'][idx,:]
                                mom.odd = templates_overview[1].data['average_moments_odd'][idx,:]
                                
             
         #∫ wverview
                        else:
                    
                            w_ = 1.
                            cc = True
                    except:
                    #    print ('not a valid measurement')
                        pass
                    if cc:
                        #print (i)
                        #mom = tab_mute.images[i].moments.get_moment(0.,0.)
                        mom = save_[i]['moments'].get_moment(0.,0.)

                        count_g +=1
                        
                        Mf = mom.even[mom.M0]
                        SN_mute = Mf/np.sqrt(save_[i]['moments'].get_covariance()[0][mom.M0,mom.M0])
              
                        mask_bool = True


          
                        
                        if limits['SN_Mf_min'] > SN_mute: 
                            #print ('not passing SN min')
                            mask_bool = False
                        if limits['SN_Mf_max'] < SN_mute: 
                            #print ('not passing SN maxx')
                            mask_bool = False
                        if limits['Mf_min'] > Mf: 
                            #print ('not passing Mf_min')
                            mask_bool = False
                        if limits['Mf_max'] < Mf: 
                            #print ('not passing Mf_max')
                            mask_bool = False
                        if mask_bool:
                                
                                #print (i)
                                t = save_[i]['moments'].make_templates( config['sigma_xy'],sigma_flux = config['sigma_flux'], sn_min= config['sn_min'], sigma_max= config['sigma_max'],sigma_step= config['sigma_step'], xy_max= config['xy_max'])
                                ID_mute = copy.copy(save_[i]['index'])
                                gc.collect()
                                try:
                                    p0 = save_[i]['p0']
                                    p0_PSF = save_[i]['p0_PSF']
                                               
                                except:
                                    
                                    p0 = 0
                                    p0_PSF = 0
              
                                if t[0] is None:
                                    continue
                                else:   
                                    for tmpl in t:
                                        count +=1
                                        tmpl.p0 = p0
                                        tmpl.p0_PSF = p0_PSF
                                        tmpl.id = ID_mute
                                        tmpl.nda *= w_/downsample_factor
                                        templates.append(tmpl)
                            #except:
                            #        pass
            
            print ('templates: ',count, ' from galaxies: ', count_g,t_index, d_index)
            save_obj(output_folder+'/templates/templates_junk/templates_{0}_{1}'.format(t_index,d_index),templates)
            print ('saved',t_index, d_index)
            #del template_list
            del save_
            gc.collect()
        #except:
        #    print ('problems with file ','(IS)_templates_'+d_index)
        



        
def make_templates(output_folder,**config):
    if config['MPI']:
        from mpi4py import MPI 
    print ('making templates')

    if ('compute' in config['stage']) or ('assembly' in config['stage']):
        
        # read list of targets :
        noise_tier = []
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
        if os.path.exists(output_folder+'/noisetiers.fits'):
            count = 0
            reading = True
            while reading:
                try:
                    NT_ = pf.open(output_folder+'/noisetiers.fits')
                    reading = False
                except:
                    time.sleep(10)
                    count += 1
                    if count > 6*5:
                        reading = False
                        sys.exit()
                        

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
        print ('runs to do [compute stage]: ',len(runs),' out of: ', len(deep_files)*len(noise_tier))
    if 'resume_templates' in config['stage']:   
       
        print ('resume templates')
        

        deep_files = glob.glob(output_folder+'/templates/'+'/t*.pkl')
        try:
            deep_files[0]
        except:
            deep_files = glob.glob(output_folder+'/templates/'+'/IS_t*.pkl')
            
        moments = []
        moments_odd = []
        error = []
        #error_odd = []
        indexu = []
        mof_index = []
        ra = []
        dec = []
        ra_DF = []
        dec_DF = []
        MAG_I = []
        tilename = []
        p0_ = []
        p0_PSF_ = []
        for ii in frogress.bar(range(len(deep_files))):

            df = deep_files[ii]
            try:
                save_ = load_obj(df.strip('.pkl'))
                for index in (save_.keys()):

                    mom = save_[index]['moments'].get_moment(0.,0.)

                    moments.append(mom.even)
                    moments_odd.append(mom.odd)
                    error.append(np.sqrt(save_[index]['moments'].get_covariance()[0].diagonal()))
                    #error_odd.append(np.sqrt(save_[index]['moments'].get_covariance()[0].diagonal()))
                    indexu.append(save_[index]['index'] )

                    try:
                        mof_index.append(save_[index]['MOF_index'] )
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
                print (df)
                
                

        
        moments = np.array(moments)
        print (moments.shape)
        # Rotate all the templates:
        phi = -0.5*(angle(moments[:,:].T))
        e_,o_ = rotate(moments[:,:].T,moments[:,:2].T,phi)
        new_DV = np.hstack([e_,o_])
        # load noisetiers ------
        nt = pf.open(output_folder+'/noisetiers.fits')

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


        # let's define a region of SN < 100.
        mask = new_DV[:,0]/np.sqrt(u[1][0,0])< 100.
        mask = (new_DV[:,0]/np.sqrt(u[-1][0,0])< 10000000.)



        mask_w,w,even_new,odd_new = select_obj_w(new_DV_[mask,:],np.array(moments)[mask,:],np.array(moments_odd)[mask,:],config['sigma_group_templates'])  
        
        
        newweigths = np.zeros(len(mask))
        newweigths = w
        
        print (len(newweigths[mask_w]),len(newweigths),np.sum(newweigths))


                    
                    
                    
                    

        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="index",format="K",array=np.array(indexu)))
        
        
        col.append(fits.Column(name="moments",format="5E",array=np.array(moments)))
        col.append(fits.Column(name="moments_odd",format="5E",array=np.array(moments_odd)))
        
        col.append(fits.Column(name="average_moments",format="5E",array=np.array(even_new)))
        col.append(fits.Column(name="average_moments_odd",format="5E",array=np.array(odd_new)))
          
        
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
        tbhdu = fits.BinTableHDU.from_columns(col)
        prihdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([prihdu,tbhdu])
        try:
            thdulist.writeto(output_folder+'/templates_overview.fits',overwrite=True)
        except:
            pass
        
        
        
        # make 

    

    if 'compute' in config['stage']:
        
    
        while run_count<len(runs):
            comm = MPI.COMM_WORLD
            print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
            if (run_count+comm.rank) < len(runs):
                pipeline(output_folder,config, deep_files, targets_properties, runs, run_count+comm.rank)
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
                    

                    
                    '''
                    
                    boia
                    '''
                    print ('total templates ',count)
                    print ('done')


                    end = timeit.default_timer()
                    print ('saving NOISETIER #',t_index, ', # templates: ',(len(tab_templates.templates))   )
                    

                    save_template(tab_templates,output_folder+'/templates_NOISETIER_{0}.fits'.format(t_index),EFFAREA,t_index)
                    #except:
                    #    pass



                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        print ('done assembly')

