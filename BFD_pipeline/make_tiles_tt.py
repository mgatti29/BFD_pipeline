from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
from ngmix.jacobian import Jacobian
import os, sys
from .read_meds_utils import Image, MOF_table, DetectionsTable,render_gal,_add_T_and_scale,save_,get_box_sizes,get_sigma_size,_make_image,initialise_entries,set_noise,setup_templates_table,select_obj,subtract_background_,check_mask_and_interpolate,mfrac_
import copy
from astropy import units as uu
from astropy.coordinates import SkyCoord
import h5py as h5
import bfd
import sxdes
import numpy as np
import math
from .utilities import  save_obj, load_obj
import astropy.io.fits as fits
import frogress
import ngmix
import meds
import shredder
import esutil as eu
import pytest
import logging
import pytest
import sxdes
import galsim
from bfd.momentcalc import MomentCovariance
import timeit
logger = logging.getLogger(__name__)


'''
General purpose
This pipeline generates 1 COADD tile for templates and/or targets, and measures their moments.
Targets and templates can be located at random or over a grid. A detection step can also be included.

In short : 
targets/templates over a grid, no detection step -> 'stamp' simulations (in the sense it has no blending and detection effects)
targets/templates at random, with detection step -> true 'tile' simulations.(it has blending and detection effects)
'''


                    
verbose = False
def pipeline_targets(config, params_image_sims, ii_chunk, do_templates = False):
    
    if do_templates:
        print ('running chunk templates -'+str(ii_chunk))
    else:
        print ('running chunk targets -'+str(ii_chunk))

    # output file **************************
    if do_templates:
        path = config['output_folder']+'/templates/'+'/IS_templates__chunk_'+str(ii_chunk)+'.pkl'
    else:
        path = config['output_folder']+'/targets/'+'ISm_targets_{0}.fits'.format(ii_chunk)
        
        
    if not os.path.exists(path):

        # this is for debugging. There's an option in the config file to also save the tiles/stamps
        if config['debug']:
            debug_images = dict()
            debug_images['det_stamps']= []
            debug_images['images_aftershredding'] = []
            debug_images['wcs_coordinates'] =[]
            debug_images['images_preshredding']= []
            debug_images['images']= []
            debug_images['mask'] = []
            debug_images['seg']= []
        
        
        try:
            
            config['perfect_deblender']
        except:
            config['perfect_deblender'] = False
            
        try:
            config['background'] 
        except:
            config['background'] = 0.
            
        try:
            config['mfrac']
        except:
            config['mfrac'] = 1.
        try:
            config['mask_data']
            
            if config['mask_data']:
                mask_data = np.load(config['mask_data'],allow_pickle=True)
                                
        except:
            config['mask_data'] = False

        
        # create 2 target tables to store coordinates/moments etc. 
        # we create 2(tab_targets & tab_targets_m) because we need a table for each of the positively/negatively sheared version of the simulated tiles.
  
        tab_targets = TargetTable(n = config['n'],sigma = config['sigma'])

        tab_targets = initialise_entries(tab_targets)
   
        tab_targets_m = copy.deepcopy(tab_targets)
   

        
        if do_templates:
            tab_detections,params_template = setup_templates_table(config)
            replicas = 1
        else:
            replicas = config['number_of_replicas']
    
            
        # n.b.: 'replicas' means that we generate multiple realisations of the tiles before saving them to disk. 
        # we set them to 1 for templates just because we need less templates usually.
            
            
            
        
        for rep in frogress.bar(range(replicas)):
       
            if verbose:
                print ('rep #',rep)
            # set the noise ************************************************************
            noise_ext = set_noise(do_templates,config)
                

            # initialise the tile ***********************************************************
            if verbose:
                print ('initialise the tile')
            tile = dict()
            for band in config['bands']:
                tile[band] = dict()
                if do_templates:
                     tile[band]['image_n'] = np.zeros((config['size_tile'],config['size_tile']))
                else:
                    tile[band]['image_p'] = np.zeros((config['size_tile'],config['size_tile']))
                    tile[band]['image_m'] = np.zeros((config['size_tile'],config['size_tile']))

            #********************************************************
            #                  INJECTION POSITIONS                  #
            #********************************************************
            # initialise the tile ***********************************************************
            if verbose:
                print ('determine positions')
            Input_catalog = dict()

            if do_templates:
                xxe = config['grid_templates']
            else:
                xxe = config['grid_targets']
                
            if  (not config['poisson']):
                spacing = math.ceil((config['size_tile']-80)/(math.ceil(np.sqrt(config['gal_per_tile']))))
                x_a = []
                y_a = []
                for i in range(math.ceil((config['size_tile']-80)/spacing)):
                    for j in range(math.ceil((config['size_tile']-80)/spacing)):
                        x_a.append(40+np.int(spacing*0.5)+spacing*i)
                        y_a.append(40+np.int(spacing*0.5)+spacing*j)
    
                x_a = np.array(x_a)[:config['gal_per_tile']]
                y_a = np.array(y_a)[:config['gal_per_tile']]
            else:
                
                

                obj_ = np.random.poisson(config['gal_per_tile'])
                if not xxe:
                    x_a = np.random.randint(40,config['size_tile']-40,obj_)
                    y_a = np.random.randint(40,config['size_tile']-40,obj_)

                else:
                    spacing = math.ceil((config['size_tile']-80)/(math.ceil(np.sqrt(obj_))))
                    x_a_f = []
                    y_a_f = []
                    for i in range(math.ceil((config['size_tile']-80)/spacing)):
                        for j in range(math.ceil((config['size_tile']-80)/spacing)):
                            x_a_f.append(40+np.int(spacing*0.5)+spacing*i)
                            y_a_f.append(40+np.int(spacing*0.5)+spacing*j)
     
                    x_a = np.array(x_a_f)[:obj_]
                    y_a = np.array(y_a_f)[:obj_]
            
            
            
            # shear positions (only for targets, templates are not sheared!) **************
            
            # INJECT BLENDS ***********************************
            # This part artificially adds blens to the tile.
            #try:
 
            if config['injecting_blends']:
                try:
                    if len(config['distance_blends']) ==2:
                        
                        x_shifts = np.random.randint(config['distance_blends'][0]/0.263,config['distance_blends'][-1]/0.263,len(x_a))
                        y_shifts = np.random.randint(config['distance_blends'][0]/0.263,config['distance_blends'][-1]/0.263,len(x_a))
                        
                        x_a = np.hstack([x_a,x_a+x_shifts])
                        y_a = np.hstack([y_a,y_a+y_shifts])
                        
                        print ('shifted')
                        
                except:
                    theta =  np.radians(np.random.randint(0,36000,len(x_a))*1./100.)
                
                    #theta =  np.radians(np.random.randint(0,10,len(x_a))*36.)
                    cos2angle = np.cos(theta)
                    sin2angle = np.sin(theta)

                    x_shifts = config['distance_blends']/0.263*cos2angle
                    y_shifts = config['distance_blends']/0.263*sin2angle

                    x_a = np.hstack([x_a,x_a+x_shifts])
                    y_a = np.hstack([y_a,y_a+y_shifts])
                        
            #except:
            #    pass
     
            # **************************************************
        
        
            if verbose:
                print ('shear positions')

            if not do_templates:
                for ll in range(2):
                
                    xc_a = []
                    yc_a = []
                    shear = galsim.Shear(g1=config['g1'][ll], g2=config['g2'][ll])

                    # this is the matrix that does shearing in u, v
                    a = shear.getMatrix()

                    for i in range(len(x_a)):
                        #print ('input  --- ', x_a[i],y_a[i])

                        # we need the canonical image center in (u, v) for undoing the
                        # shearing

                        row_cen = x_a[i]
                        col_cen = y_a[i]


                        jac = ngmix.jacobian.Jacobian(row=  config['size_tile']/2.,
                              col =  config['size_tile']/2.,
                              dudrow=0.,
                              dudcol=0.263,
                              dvdrow=0.263,
                              dvdcol=0.)

                        v_cen, u_cen = jac.get_vu(row=config['size_tile']/2., col=config['size_tile']/2.)

                        # apply WCS to get to world coords
                        v, u = jac.get_vu(row=row_cen, col=col_cen)

                        # unshear (subtract and then add the canonical center in u, v)
                        u = np.atleast_1d(u) - u_cen
                        v = np.atleast_1d(v) - v_cen

                        #print (u,v)

                        pos = np.vstack((u, v))

                        out = np.dot(a, pos)
                        assert out.shape[1] == u.shape[0]
                        u_sheared = out[0, :] + u_cen
                        v_sheared = out[1, :] + v_cen

                        rows_sheared, cols_sheared = jac.get_rowcol(v=v_sheared, u=u_sheared)

                        xc_a.append(rows_sheared[0])
                        yc_a.append(cols_sheared[0])

                    if ll == 0:
                        x_a_p = (np.array(xc_a))
                        y_a_p = (np.array(yc_a))                  
                    elif ll == 1:
                        x_a_m = (np.array(xc_a))
                        y_a_m = (np.array(yc_a))

                    

    
            
            
            AREA =  (config['size_tile']-80)**2
            
            if verbose:
                print ('')    
                print ('')    
                print ('*****************')
                print ('tile description')
                print (' # obj: ',obj_,len(x_a))
                print ('coordinates')
                print (x_a)
                print (y_a)
                print ('*****************')
            
            x_a_poisson_selection = np.random.randint(40,config['size_tile']-40,1)[0]
            y_a_poisson_selection = np.random.randint(40,config['size_tile']-40,1)[0]

            
            
            
            
            
            # REMOVE CLOSE PAIRS **************************************************
            '''
            This snippet remove close pairs.
            the_same = True
            count_t = 0

            while the_same:

                indexes_final = np.arange(len(x_a))
                catalog = SkyCoord(ra=x_a*uu.arcsec*0.263, dec=y_a*uu.arcsec*0.263)  
                idx, d2d, d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=2) 
                dist_pix = np.sqrt((x_a-x_a[idx])**2+(y_a- y_a
[idx])**2)*0.263
                # creating index pairs

                dist_t = 2.
                # unique pairs of close obj.
                vv = np.vstack([idx[dist_pix<dist_t],np.arange(len(idx))[dist_pix<dist_t]]).T
                vv_ = []
                if len(vv)>2:
                    for v in vv:
                        vv_.append(np.sort(v))
                    idx_close_pairs = np.unique(np.array((vv_))[:,0])
                else:
                    idx_close_pairs=-1

                indexes_unique = ~np.in1d(np.arange(len(x_a)),idx_close_pairs)

                indexes_final[indexes_final] = indexes_final[indexes_final] & indexes_unique

                if (len(x_a)==len(x_a[indexes_unique])):
                    the_same = False
                else:
                    x_a = x_a[indexes_unique]
                    y_a = y_a[indexes_unique]
                count_t+=1
            '''
            
            

    
            
            '''
            STAMP sims:
                sky_density =  1./templates_tot (for templates)
                area of non detection: 1.


            POISSON SIMS sims:
                sky_density = gal_per_tile/area_tile * 1./ templates_tot ~ 1/ area_tot (deep fields) (for templates)
                area_for_pseudo_target = area_of_tile (if tile is homoegenous) # 1 per tile.
            
            '''

      
            # this sets the PSF for the til e**********************
            pp = list(params_image_sims.keys())
  
            if config['PSF_TYPE']=='turb':
                psf_fwhm = config['turb'][0]+(np.random.random(1)*config['turb'][1])[0]
                Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)
                psf_pars_ = {'pars':[.0, 0., 0., 0., Tpsf, 1.0],'turb':True}

                
            #********************************************************
            #                  GENERATE GALAXIES.                   #
            #********************************************************
            if verbose:
                print ('generate galaxies')
            for real in range(len(x_a)):
                if verbose:
                    print ('gal #',real)
                #try:
                if 1==1:
                    
                    # make WCS for the object ***********************
                    cent=(y_a[real],x_a[real])
                    origin = (0.,0.)
                    duv_dxy = np.array([[0.263, 0.],
                                        [0., 0.263]])
                    wcs_ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)

                    if not do_templates:
                        cent=(y_a_p[real],x_a_p[real])
                        wcs_p = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)    
                        cent=(y_a_m[real],x_a_m[real])
                        wcs_m = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)    
                                      
                        
                    # choose galaxy model *****************
                    redoit = True
                    count_repeat = 0 
                    while redoit:
                        if verbose:
                            print ('repeat #',count_repeat)
                        
                        
                        p0 = pp[np.random.randint(0,len(pp),1)[0]] 

                        if do_templates: 
                            # if poisson it must be random *****
                            if config['poisson']:
                                p0 = pp[(np.random.randint(0,len(pp),1)[0])%len(pp)]
                            else:
                                p0 = pp[(config['gal_per_tile']*ii_chunk+real)%len(pp)]
                                
          
                        for b, band in enumerate(config['bands']):
                            gp = params_image_sims[p0][band]['gal_pars']
                            
                            if (gp[4]> config['size_treshold']):
                                #skipe this model because it's too big.
                                #skip_this_model+=1. 
                                
                                if b == 0:
                                    redoit= True
                                else:
                                    redoit = redoit | True
                                
                            else:
                                if b == 0:
                                    redoit= False
                                else:
                                    redoit = redoit | False
                  
                    if verbose:
                        print ('got model')
                        
                    # read the galaxy and PSF and randomly rotate the galaxy
                    twotheta = 2.0 * np.radians(np.random.randint(0,36000)*1./100.)
                    cos2angle = np.cos(twotheta)
                    sin2angle = np.sin(twotheta)
                    galaxy_info = dict()

                    for b in config['bands']:
                        rendering = True
                        count_rendering = 0
                        while rendering:
                            gp = copy.deepcopy(params_image_sims[p0][b]['gal_pars'])
              
                            if config['PSF_TYPE']=='turb':
                                psfp = copy.copy(psf_pars_)
                                p0_PSF = np.int(psfp['pars'][4]*100000)

                                
                            mm1= gp[2] * cos2angle +  gp[3] * sin2angle
                            mm2= -gp[2] * sin2angle + gp[3] * cos2angle
                            gp[2] = copy.copy(mm1)
                            gp[3] = copy.copy(mm2)

                            if not do_templates:
                                galaxy_info[b] = {'gal_p':gp,'psf_p':psfp,'wcs':wcs_,'wcs_m':wcs_m,'wcs_p':wcs_p,'p0':p0, 'p0_PSF':p0_PSF,'x_a_p':x_a_p[real],'y_a_p':y_a_p[real],'x_a_m':x_a_m[real],'y_a_m':y_a_m[real]}
                                
                            else:
                                galaxy_info[b] = {'gal_p':gp,'psf_p':psfp,'wcs':wcs_,'p0':p0, 'p0_PSF':p0_PSF,'x_a':x_a[real],'y_a':y_a[real]}
                            
                            if verbose:
                                print ('pre rendering')
                            if do_templates:
                                mute_p,simulated_psf,jac = render_gal(gp,psfp,wcs_,config['size_tile'], g1 = 0., g2 = 0.,return_PSF=True)
                                mute_m = 1.
                            else:
                                mute_p,simulated_psf,jac = render_gal(gp,psfp,wcs_p,config['size_tile'], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)
                                mute_m,jac = render_gal(gp,psfp,wcs_m,config['size_tile'],  g1 = config['g1'][1], g2 = config['g2'][1])
                                
                                                    
                            if verbose:
                                print ('after rendering')
                            sv = False
                            if (mute_p is not None) and (mute_m is not None):
                                if do_templates:
                                    tile[b]['image_n'] += mute_p
                                else:
                                    tile[b]['image_p'] += mute_p
                                    tile[b]['image_m'] += mute_m
                                sv = True
                                rendering = False
                            else:
                                count_rendering += 1
                                if count_rendering>2:
                                    rendering = False
                                    print ('model rendering failes somehow')
                        if sv:
                            
                             Input_catalog[real] = galaxy_info    
                #except:
                
                #    if (gp[4]> config['size_treshold']):
                #        pass
                #    else:
                #        print ('problems with model ',p0)
                        
                        
            if verbose:        
                print ('PSF ', p0_PSF)
                print ('len catalog ',len(Input_catalog.keys()))
            
            
            
            
            #********************************************************
            #                       ADD NOISE                       #
            #********************************************************
            
            for b in config['bands']:
                if noise_ext:
                        noise = np.random.normal(size = (config['size_tile'],config['size_tile']))*noise_ext
                        tile[b]['noise_level']  = noise_ext

                else:
                        print ('NOT SUPPORTED')
                        sys.exit()
                        try:
                            noise = np.random.normal(size = (config['size_tile'],config['size_tile']))*np.sqrt(1.0/tile[band]['weight'])
                        except:
                            noise = np.random.normal(size = (config['size_tile'],config['size_tile']))*np.sqrt(1.0/np.median(tile[band]['weight']))  
                tile[b]['noise'] = noise


                if not config['noiseless']:
                    if do_templates:
                        tile[b]['image_n_noisefree'] = copy.deepcopy(tile[b]['image_n'])
                        tile[b]['image_n']+=tile[b]['noise']
                        
                    else:
                        tile[b]['image_p_noisefree'] = copy.deepcopy(tile[b]['image_p'])
                        tile[b]['image_m_noisefree'] = copy.deepcopy(tile[b]['image_m'])
                        tile[b]['image_p']+=tile[b]['noise']
                        tile[b]['image_m']+=tile[b]['noise']
                      

                            # apply mask
                tile[band]['mask'] = np.zeros_like(noise).astype(np.int)
  
                          
                        
            #********************************************************
            #               sEXTRACTOR DETECTION                    #
            #********************************************************
            
            if config['mode_detection'] == 'detection':
                detection_cat = dict()
                seg_map = dict()
                for b in config['bands']:
                    detection_cat[b] = dict()
                    seg_map[b] = dict()
                    if do_templates:
                        detection_cat[b]['image_n'], seg_map[b]['image_n'] = sxdes.run_sep(tile[band]['image_n'], noise_ext)
                    else:
                        detection_cat[b]['image_p'], seg_map[b]['image_p'] = sxdes.run_sep(tile[band]['image_p'], noise_ext)
                        detection_cat[b]['image_m'], seg_map[b]['image_m'] = sxdes.run_sep(tile[band]['image_m'], noise_ext)

        
        
        
                    #********************************************************
                    #               RUN SHREDDER DEBLENDER                  #
                    #********************************************************
                    
                    if config['shredder']:
                        
                       #   print ('pre shredding')
                        st = timeit.default_timer()
                        shredder_cat = dict()
                        seed = 113
                        rng = np.random.RandomState(seed)
                        guess_model = 'dev'

                        wmp = np.ones((config['size_tile'],config['size_tile']))*1./tile[b]['noise_level']**2
                        jac_shredder = ngmix.jacobian.Jacobian(row= config['size_tile']/2.,
                                              col= config['size_tile']/2.,
                                              dudrow=wcs_.jac[0,1],
                                              dudcol=wcs_.jac[0,0],
                                              dvdrow=wcs_.jac[1,1],
                                              dvdcol=wcs_.jac[1,0])


                        cent=(config['size_tile']/2.,config['size_tile']/2.)
                        origin = (0.,0.)
                        duv_dxy = np.array([[0.263, 0.],
                                            [0., 0.263]])
                        wcs_ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)
                        
                        # SIMULATE PSF
                        _,simulated_psf,_ = render_gal(gp,psfp,wcs_,config['size_tile'], g1 = 0., g2 = 0.,return_PSF=True)


        
                        psf_o = ngmix.Observation(simulated_psf,
                                        jacobian=jac_shredder)


                        
                        if do_templates:
                            loop_keys = ['image_n']
                        else:
                            loop_keys = ['image_p','image_m']
                
                        for im_type in loop_keys:
                            shredder_cat[im_type] = dict()
                     
                            mbobs_ = ngmix.Observation(tile[b][im_type],
                                    weight=wmp,
                                    meta={"orig_row": (config['size_tile'])/2., "orig_col": (config['size_tile'])/2.},
                                    jacobian=jac_shredder,
                                    psf=psf_o)


                            #
                            mbobs_ = ngmix.observation.get_mb_obs(mbobs_)

                            dt = np.dtype([('shift', '<f4', (2,)),('flux', 'f4'), ('hlr', 'f4'),('col', 'f4'),('row', 'f4')])

                        
                            m = []
                            for i in range(len(detection_cat[b][im_type]['flux'])):
                                m.append(((0.,0.),detection_cat[b][im_type]['flux'][i],
                            detection_cat[b][im_type]['flux_radius'][i],
                            detection_cat[b][im_type]['x'][i],
                            detection_cat[b][im_type]['y'][i]))
                            d_ = np.array([m],dtype=dt)

                            objs_ = _add_T_and_scale(d_[0],0.263)#0.263) # 0.263);
         
                            gm_guess = shredder.get_guess(
                                objs_,
                                jacobian=mbobs_[0][0].jacobian,
                                model=guess_model,
                                rng=rng,
                            )
              
                            s = shredder.Shredder(obs=mbobs_, psf_ngauss=2, rng=rng)
                            s.shred(gm_guess)
                            res = s.get_result()
                            

                            models = s.get_model_images()
                            shredder_cat[im_type]['models'] = models
                            shredder_cat[im_type]['res'] = res
                        end = timeit.default_timer()
           
              
                        
            
            #********************************************************
            #                  MEASURE  MOMENTS                     #
            #********************************************************
            
            imc_keys = np.array(list(Input_catalog.keys()))
            
            def get_coordinates(Input_catalog,im_type):
                ll = []
                for rel in Input_catalog.keys():
                    if im_type == 'image_p':
                        ll.append([Input_catalog[rel][b]['y_a_p'],Input_catalog[rel][b]['x_a_p']])
                    elif im_type == 'image_m':
                        ll.append([Input_catalog[rel][b]['y_a_m'],Input_catalog[rel][b]['x_a_m']])
                    elif im_type == 'image_n':
                        ll.append([Input_catalog[rel][b]['y_a'],Input_catalog[rel][b]['x_a']])
                return np.array(ll)
                
                


            if do_templates:
                loop_keys = ['image_n']
            else:
                loop_keys = ['image_p','image_m']
                
                
            
            # we want to measure moments at the location of the detctions for Poisson-like sims, or at the injection coordinates for stamp-like sims.
            final_coordinates= dict()
            for im_type in loop_keys:
                
                ll = get_coordinates(Input_catalog,im_type)
                
                if config['mode_detection'] == 'detection':
                    extra_obj_data_fields = [('number', 'i8'),]
                    obj_data = meds.util.get_meds_input_struct(len(detection_cat[config['bands'][0]][im_type]['x']), extra_fields=extra_obj_data_fields)
                    obj_data["id"] = detection_cat[config['bands'][0]][im_type]['number']
                    obj_data["number"] = detection_cat[config['bands'][0]][im_type]['number']
                    obj_data["ra"] = detection_cat[config['bands'][0]][im_type]['x']
                    obj_data["dec"] = detection_cat[config['bands'][0]][im_type]['y']
                    obj_data["box_size"] = get_box_sizes(detection_cat[config['bands'][0]][im_type])

                    len_loop = len(detection_cat[config['bands'][0]][im_type]['y'])
                    if verbose:
                        print ('len detections: ',len_loop)
                else:
                    len_loop = len(Input_catalog.keys())
                    
        
                # THIS GROUPS TEMPLATES/TARGETS TOGETHER WITHIN A GIVEN RADIUS ************************************+***************+***************+***************+*
                if config['mode_detection'] == 'detection':
                    indexes_final = np.array(detection_cat[config['bands'][0]][im_type]['x'] == detection_cat[config['bands'][0]][im_type]['x'])
                else:
                    ll = get_coordinates(Input_catalog,im_type)
                    indexes_final = np.array(list(Input_catalog.keys()))[:] == np.array(list(Input_catalog.keys()))[:]
                    #print (indexes_final)
                    #indexes_final = np.array(list(Input_catalog.keys()))[:,0] == np.array(list(Input_catalog.keys()))[:,0]
               
               
            
                #if do_templates: #MG: YOU WANT TO CHANGE THIS ALSO FOR TARGETS I'D SAY -----
                if 1==1:
                    if config['mode_detection'] == 'input':
                        ll = get_coordinates(Input_catalog,im_type)
                        #print (ll.shape)
                        x = ll[:,0]
                        y = ll[:,1]
                        
                 

                    if config['mode_detection'] == 'detection':
                        x = detection_cat[config['bands'][0]][im_type]['x']
                        y = detection_cat[config['bands'][0]][im_type]['y']

                    
                    mask_too_close = select_obj(x,y,config['radius_blends_templates']/0.263)
                    final_coordinates[im_type] = [x[mask_too_close],y[mask_too_close]]
                    indexes_final[~mask_too_close] = False
                    
                    
                    mask_too_close_= select_obj(x[mask_too_close],y[mask_too_close],config['radius_blends_templates']/0.263)
                    #print ('')
                    #print ([x,y])
                    #print ([x[mask_too_close],y[mask_too_close]])
                    #print ('checkremove: ',len(mask_too_close[~mask_too_close]))
                    
                    
                    
                    
                    
                
                    '''
                    the_same = True
                    count_t = 0
                    while the_same:
                        catalog = SkyCoord(ra=x*uu.arcsec*0.263, dec=y*uu.arcsec*0.263)  
                        idx, d2d, d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=2) 
                        dist_pix = np.sqrt((x-x[idx])**2+(y-y[idx])**2)*0.263
                        # creating index pairs

                        dist_t = config['radius_blends_templates']
                        # unique pairs of close obj.
                        vv = np.vstack([idx[dist_pix<dist_t],np.arange(len(idx))[dist_pix<dist_t]]).T
                        vv_ = []
                        if len(vv)>2:
                            for v in vv:
                                vv_.append(np.sort(v))
                            idx_close_pairs = np.unique(np.array((vv_))[:,0])
                        else:
                            idx_close_pairs=-1

                        indexes_unique = ~np.in1d(np.arange(len(x)),idx_close_pairs)

                        indexes_final[indexes_final] = indexes_final[indexes_final] & indexes_unique

                        if (len(x)==len(x[indexes_unique])):
                            the_same = False
                        else:
                            x = x[indexes_unique]
                            y = y[indexes_unique]
                        count_t+=1
                        if count_t>100:
                            print ('stuck')

                
                    final_coordinates[im_type] = [x,y]

                    '''
                #print ('final coordinates: ',len(np.arange(len_loop)[indexes_final]))
                #**************************************
                #           MEASUREMENT LOOP          #
                #**************************************
                
                templates_id = []
             
                ix_ =0
               
                for ix in np.arange(len_loop)[indexes_final]:
                        
                        # the outer 40 pixels are never used to inject galaxies, so if there's a detection there it shoud be dropped. 
                        if config['mode_detection'] == 'detection':
                            obj_within_good_area = (detection_cat[config['bands'][0]][im_type]['x'][ix]>=30) & (detection_cat[config['bands'][0]][im_type]['x'][ix]<= (config['size_tile']-30)) &  (detection_cat[config['bands'][0]][im_type]['y'][ix]>=30) & (detection_cat[config['bands'][0]][im_type]['y'][ix]<= (config['size_tile']-30))
                        else:
                            obj_within_good_area = True
                            
                      
                        if obj_within_good_area:
                            images_a = []
                            wcs_a = []
                            psf_a = []
                            bands_a = []
                            noise_a = []

                            for band in config['bands']:
                                # cut the image **************************


                                if config['mode_detection'] == 'detection':
                                    box_size = obj_data["box_size"][ix]
                                    #print ('detection')
                                    #print (detection_cat[config['bands'][0]][im_type]['x'][ix],detection_cat[config['bands'][0]][im_type]['y'][ix])
                                    maskx = (np.arange(config['size_tile'])>=detection_cat[config['bands'][0]][im_type]['x'][ix]-box_size/2) & (np.arange(config['size_tile'])<detection_cat[config['bands'][0]][im_type]['x'][ix]-box_size/2+box_size)
                                    masky = (np.arange(config['size_tile'])>=detection_cat[config['bands'][0]][im_type]['y'][ix]-box_size/2) & (np.arange(config['size_tile'])<detection_cat[config['bands'][0]][im_type]['y'][ix]-box_size/2+box_size) 



                                    #match to input catalog to find PSF & noise model for the stamp *******
                                    goldcat = SkyCoord(ra=[detection_cat[config['bands'][0]][im_type]['x'][ix]*uu.degree*0.263/60.], dec=[detection_cat[config['bands'][0]][im_type]['y'][ix]*uu.degree*0.263/60.])  
                                    catalog = SkyCoord(ra=ll[:,0]*uu.degree*0.263/60., dec=ll[:,1]*uu.degree*0.263/60.)  
                                    idx, d2d, d3d = goldcat.match_to_catalog_sky(catalog, nthneighbor=1) 
                                    dist_pix = np.sqrt((detection_cat[config['bands'][0]][im_type]['x'][ix]-ll[idx][0][0])**2+(detection_cat[config['bands'][0]][im_type]['y'][ix]-ll[idx][0][1])**2)
                                    
                                    
                                    

                                    #print ('distance detection from nearest input ',dist_pix)
                                    psf_p =  Input_catalog[imc_keys[np.int(idx)]][band]['psf_p']
                                    gal_p =  Input_catalog[imc_keys[np.int(idx)]][band]['gal_p']
                                    
                                    # need to substitute xy0 with detection coordinates.
                                    
                                    
                                    cent=(detection_cat[config['bands'][0]][im_type]['x'][ix],detection_cat[config['bands'][0]][im_type]['y'][ix])
                                    origin = (0.,0.)
                                    duv_dxy = np.array([[0.263, 0.],
                                                    [0., 0.263]])
                                    wcs_ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)
                                
                                
                                   # wcs_ = copy.copy(Input_catalog[imc_keys[np.int(idx)]][band]['wcs'])
                                    wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0])
                                    p0 = Input_catalog[imc_keys[np.int(idx)]][band]['p0']
                                    
                                    # basically if the detection is too far away from the input object, we flag this as a spurious object

                                else:
                                    maskx = (np.arange(config['size_tile'])>=ll[ix][0]-20) & (np.arange(config['size_tile'])<ll[ix][0]-20+40)
                                    masky = (np.arange(config['size_tile'])>=ll[ix][1]-20) & (np.arange(config['size_tile'])<ll[ix][1]-20+40) 


                                    psf_p = Input_catalog[imc_keys[np.int(ix)]][band]['psf_p']
                                    gal_p = Input_catalog[imc_keys[np.int(ix)]][band]['gal_p']
                                    
                                    if im_type == 'image_p':
                                        wcs_ = copy.copy(Input_catalog[imc_keys[np.int(ix)]][band]['wcs_p'])
                                    elif im_type == 'image_m':
                                        wcs_ = copy.copy(Input_catalog[imc_keys[np.int(ix)]][band]['wcs_m'])                   
                                    elif im_type == 'image_n':
                                        wcs_ = copy.copy(Input_catalog[imc_keys[np.int(ix)]][band]['wcs'])      
                                        
                                    
                                    
                                    wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0])

                                    p0 = Input_catalog[imc_keys[np.int(ix)]][band]['p0']


                                seg_ = seg_map[b][im_type][masky,:][:,maskx]
                                
                                if not do_templates:
                                    image_stamp = tile[band][im_type][masky,:][:,maskx]+config['background']
                                else:
                                    image_stamp = tile[band][im_type][masky,:][:,maskx]
                                    
                                
                                
                                if config['mask_data']:
                                    print ('DOING MASKING STUFF')
                                    mute =  mask_data[np.random.randint(0,len(mask_data),1)[0]]
                                    mask_ =(np.zeros_like(image_stamp)).astype(np.int)
                                    mask_[:mute.shape[0],:mute.shape[1]] = mute[:mask_.shape[0],:mask_.shape[1]] 
                                    image_stamp[mask_ != 0 ] = 0
                                else:
                                    mask_ = (np.zeros_like(image_stamp)).astype(np.int)
                                    
                                
                                wt_stamp = tile[band][im_type][masky,:][:,maskx]
                                
                              
                                if config['debug']:
                                    debug_images['images_preshredding'].append(copy.deepcopy(image_stamp)) 
                                    debug_images['seg'].append(copy.deepcopy(seg_)) 
                                    
                                    
                                    
                                    
                                if config['background_subtraction'] == True:
                                    bkg,len_v = subtract_background_(copy.deepcopy(image_stamp), seg_, mask_)
                                    #print (subtract_background_(image_stamp, seg_, mask_),subtract_background_(tile[band][im_type][masky,:][:,maskx], seg_, mask_))
                                else:
                                    bkg = 0.  
                                    len_v = 1000000000.
                                
                         
                                if not do_templates:
                                    #bkg = config['background']
                                    #len_v = 1000000000.
                                    image_stamp -=bkg
                                    
                                    
                                    #bkg = subtract_background_(image_stamp, seg_, mask_)
                                   # print (bkg)
                                    
                                    
                                # SHREDDER SUBTRACTION *********
                                if (config['mode_detection'] == 'detection') and (config['shredder']):
                                    st = timeit.default_timer()
                                    
                                    pa_= (shredder_cat[im_type]['res']['band_gmix_convolved'][0].get_data())
                                    x_,y_ = pa_['col']/0.263+config['size_tile']/2.,pa_['row']/0.263+config['size_tile']/2.
                                    catalog_input = SkyCoord(ra=x_*uu.arcsec*0.263, dec=y_*uu.arcsec*0.263)  
                                    catalog_1 = SkyCoord(ra=[detection_cat[config['bands'][0]][im_type]['x'][ix]]*uu.arcsec*0.263, dec=[detection_cat[config['bands'][0]][im_type]['y'][ix]]*uu.arcsec*0.263)  
                                    min_scale = max([0.5,config['radius_blends_templates']])
                                    idx_, Aidx, d2d, d3d = catalog_input.search_around_sky(catalog_1, min_scale*uu.arcsec)
                                    pars = np.zeros(len(Aidx) * 6)
                                    beg = 0
                                    for i in Aidx:
                                        pars[beg + 0] = pa_["p"][i]
                                        pars[beg + 1] = pa_["row"][i]
                                        pars[beg + 2] = pa_["col"][i]
                                        pars[beg + 3] = pa_["irr"][i]
                                        pars[beg + 4] = pa_["irc"][i]
                                        pars[beg + 5] = pa_["icc"][i]
                                        beg += 6
                                    gm = ngmix.GMix(pars=pars)
                                    im0 = gm.make_image((config['size_tile'],config['size_tile']), jacobian=jac_shredder)
                            
         
                                    image_stamp = image_stamp-(shredder_cat[im_type]['models'][0]-im0)[masky,:][:,maskx]
                                    
                                if (config['mode_detection'] == 'detection') and (config['perfect_deblender']):
                                    
                                    
                                    
                                    # try to add the extact deblender ++++++++++
                                    min_scale = max([0.5,config['radius_blends_templates']])
                                    # cut the noise free models according to the stamp position--
                                    models_ = tile[b][im_type+'_noisefree'][masky,:][:,maskx]
                                    x_t = detection_cat[config['bands'][0]][im_type]['x'][ix]
                                    y_t = detection_cat[config['bands'][0]][im_type]['y'][ix]
                                    
                            
                                    count_blends = 0
                                    for rel in Input_catalog.keys():
                                        if im_type == 'image_p':
                                            xm, ym = Input_catalog[rel][b]['y_a_p'],Input_catalog[rel][b]['x_a_p']
                                        elif im_type == 'image_m':
                                            xm, ym = Input_catalog[rel][b]['y_a_m'],Input_catalog[rel][b]['x_a_m']
                                        elif im_type == 'image_n':
                                            xm, ym = Input_catalog[rel][b]['y_a'],Input_catalog[rel][b]['x_a']
                                        
                                        # check if coordinates are within the minimum radius to the detection: ++++++
                                        dxt = (xm-x_t)
                                        dyt = (ym-y_t)
                                        d_ = np.sqrt(dxt**2 +dyt**2)*0.263
                                        
                                        
                                        if d_ < min_scale:
                                            
        
                                            if im_type == 'image_n':
                                                wcs_f = Input_catalog[rel][b]['wcs']
                    
                                                #cent=(y_a_p[real],x_a_p[real])
                                    
                                                #cent=(detection_cat[config['bands'][0]][im_type]['x'][ix],detection_cat[config['bands'][0]][im_type]['y'][ix])
                                
                    
                                                wcs_f.xy0 = np.array([xm, ym ]) - np.array([np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0]])
                                                model_,  _,jac = render_gal(Input_catalog[rel][b]['gal_p'],Input_catalog[rel][b]['psf_p'],wcs_f,image_stamp.shape[0], g1 = 0., g2 = 0.,return_PSF=True)
                                            elif im_type == 'image_p':
                                                wcs_f = Input_catalog[rel][b]['wcs_m']
                                                wcs_f.xy0 =  np.array([xm, ym ]) - np.array([np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0]])
                                                model_,  _,jac = render_gal(Input_catalog[rel][b]['gal_p'],Input_catalog[rel][b]['psf_p'],wcs_f,image_stamp.shape[0], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)
                                            elif im_type == 'image_m':
                                                wcs_f = Input_catalog[rel][b]['wcs_p']
                                                wcs_f.xy0 =  np.array([xm, ym ]) - np.array([np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0]])
                                                model_,  _,jac = render_gal(Input_catalog[rel][b]['gal_p'],Input_catalog[rel][b]['psf_p'],wcs_f,image_stamp.shape[0], g1 = config['g1'][1], g2 = config['g2'][1],return_PSF=True)
                                            
                                            models_ -= model_
                                            count_blends +=1

                                    image_stamp = image_stamp-models_
                                                                               
                                    
                                    end = timeit.default_timer()
                             
                            
           

                                # render galaxy and psf model *****
                                cent=(image_stamp.shape[0]/2.,image_stamp.shape[0]/2.)
                                origin = (0.,0.)
                                duv_dxy = np.array([[0.263, 0.],
                                                [0., 0.263]])
                                wcs__ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)
 
                                
                                if do_templates:
                                    _,  psf_image,jac = render_gal(gal_p,psf_p,wcs__,image_stamp.shape[0], g1 = 0., g2 = 0.,return_PSF=True)
                                else:
                                    _,  psf_image,jac = render_gal(gal_p,psf_p,wcs__,image_stamp.shape[0], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)

                            
                            

                                mask_frac =0.
                                if config['interpolate_mask']:
                                    if np.sum(mask_.flatten())>0:
                                        # compute mask frac ----
                                        mask_frac = mfrac_(mask_,image_stamp.shape[0],wcs__)
                                        if mask_frac<config['mfrac']/2.:
                                            image_stamp = quick_mask_interp(copy.deepcopy(image_stamp),copy.deepcopy(mask_),noise_ext)
                                        
                                           # image_stamp, nbad = check_mask_and_interpolate(copy.deepcopy(image_stamp),copy.deepcopy(mask_))
                                
                                
                                     
                                                       
                                    
                                    
                                #print (image_stamp.shape,psf_image.shape)
                                if config['debug']:
                                    debug_images['images_aftershredding'].append([image_stamp])
                                    debug_images['mask'].append([copy.deepcopy(mask_)])
                                   # debug['bkg'] = [bkg]
                                    #debug_images['images_aftershredding'].append(psf_image)
                                    debug_images['wcs_coordinates'].append(wcs_.xy0)
                                
                                #debug_images['images'].append(image_stamp)
                                images_a.append(image_stamp)
                                wcs_a.append(wcs_)
                                psf_a.append(psf_image)
                                bands_a.append(band)
                                noise_a.append(tile[band]['noise_level'])

                            kds = bfd.multiImage(images_a, (0,0), psf_a, wcs_a, pixel_noiselist = noise_a, bandlist = bands_a ,pad_factor= config['pad_factor'])
                            wt = mc.KSigmaWeight(sigma = config['sigma']) 
                            mul = bfd.MultiMomentCalculator(kds, wt, bandinfo = config['band_dict_code'])
                            xyshift, error,msg = mul.recenter()
                            moments = mul
                            
                            mom = moments.get_moment(0,0)
                            #if mom.even[0]< 1000:
                            #    print ('Moments: ',ix,im_type, mom.even[0],wcs_.xy0,wcs__.xy0)
                            
                            mom,meb_= moments.get_moment(0,0,returnbands=True)
                            meb = np.array([m_.even for m_ in meb_])
                            
                            covm_even,covm_odd,covm_even_all,covm_odd_all = moments.get_covariance(returnbands=True)

                            psfn_ = np.zeros_like(psf_a[0])
                            psfn_[len(psfn_[0])//2,len(psfn_[0])//2] = 1
                            
                            kds_PSF = bfd.multiImage(psf_a, (0,0), [psfn_]*len(psf_a), wcs_a, pixel_noiselist = noise_a, bandlist = bands_a,pad_factor=config['pad_factor'])
                            mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, bandinfo = config['band_dict_code'])

                            if config['mode_detection'] == 'detection':
                                newcent = [detection_cat[config['bands'][0]][im_type]['x'][ix],detection_cat[config['bands'][0]][im_type]['y'][ix]]
                                newcent_shift =xyshift/0.263 #the xyshift accounts already for the jacobian / pixel scale.
                                    
                            else:
                                newcent = np.array([ll[np.int(ix)][0],ll[np.int(ix)][1]])
                                newcent_shift = xyshift/0.263

                            covgal = covm_even,covm_odd
                            covgal_per_band = covm_even_all,covm_odd_all 
                            if covgal_per_band is not None:

                                cov_even_save_per_band = []
                                cov_odd_save_per_band = []
                                for ii in range(covgal_per_band[0].shape[0]):
                                    cov_even_save_per_band.extend(covgal_per_band[0][ii][ii:])
                                for ii in range(covgal_per_band[1].shape[0]):
                                    cov_odd_save_per_band.extend(covgal_per_band[1][ii][ii:])


                            if do_templates:
                                if config['mode_detection'] == 'detection':
                                    goldcat = SkyCoord(ra=[detection_cat[config['bands'][0]][im_type]['x'][ix]*uu.degree*0.263/60.], dec=[detection_cat[config['bands'][0]][im_type]['y'][ix]*uu.degree*0.263/60.])  
                                    catalog = SkyCoord(ra=ll[:,0]*uu.degree*0.263/60., dec=ll[:,1]*uu.degree*0.263/60.)  
                                    idx, d2d, d3d = goldcat.match_to_catalog_sky(catalog, nthneighbor=1) 
                                    
                                    Wide_g = Image(Input_catalog[imc_keys[np.int(idx)]][band]['p0'], meds = [], bands = config['bands'])
                                else:
                                    Wide_g = Image(Input_catalog[imc_keys[np.int(ix)]][band]['p0'], meds = [], bands = config['bands'])


                                    
                                
                                tab_detections.add_image(Wide_g)
                               
                                tab_detections.images[ix_].moments = mul

                                mom = tab_detections.images[ix_].moments.get_moment(0.,0.)

                                
                                
                                if config['mode_detection'] == 'detection':
                                    if dist_pix >3:
                                        tab_detections.images[ix_].p0 = -1.
                                    else:
                                        tab_detections.images[ix_].p0 = Input_catalog[imc_keys[np.int(idx)]][band]['p0']
                                    tab_detections.images[ix_].p0_PSF = Input_catalog[imc_keys[np.int(idx)]][band]['p0_PSF']
                                    templates_id.append(Input_catalog[imc_keys[np.int(idx)]][band]['p0'])
                                if config['mode_detection'] == 'input':
                                    tab_detections.images[ix_].p0 = Input_catalog[imc_keys[np.int(ix)]][band]['p0']
                                    tab_detections.images[ix_].p0_PSF = Input_catalog[imc_keys[np.int(ix)]][band]['p0_PSF']
                                    templates_id.append(Input_catalog[imc_keys[np.int(ix)]][band]['p0'])
                                ix_ += 1

                            if mask_frac<config['mfrac']/2.:
                                if not do_templates:
                                    if meb[0,0] != meb[0,0]:
                                        pass
                                        #print ('NAN ',im_type,dist_pix,newcent,wcs_.xy0 )

                                    if im_type == 'image_p':
                                        if (mom.even ==  mom.even)[0]:
                                            tab_targets.add(mom, xy=newcent,id=ix,covgal=MomentCovariance(covgal[0],covgal[1]))
                                            tab_targets.p0.append(p0)
                                            tab_targets.p0_PSF.append(p0_PSF)
                                            tab_targets.ra.append(newcent[0])
                                            tab_targets.dec.append(newcent[1])
                                            tab_targets.ra_shift.append(newcent_shift[0])
                                            tab_targets.dec_shift.append(newcent_shift[1])
                                            tab_targets.AREA.append(0.)
                                            tab_targets.bkg.append(bkg)
                                            tab_targets.len_v.append(len_v)
                                            
                                            

                                            try:
                                                tab_targets.is_it_a_blend.append(count_blends)
                                            except:
                                                pass
                                            #meb_ = np.array([m_.even for m_ in meb])
                                            tab_targets.meb.append(meb[0,:])
                                            try:
                                                tab_targets.true_fluxes.append(fluxes)
                                            except:
                                                pass
                                            Mf,Mr,M1,M2,_ = mul_PSF.get_moment(0.,0.).even
                                    #

                                            tab_targets.psf_Mf.append(Mf)
                                            tab_targets.psf_Mr.append(Mr)
                                            tab_targets.psf_M1.append(M1)
                                            tab_targets.psf_M2.append(M2)

                                            tab_targets.cov_odd_per_band.append(cov_odd_save_per_band)
                                            tab_targets.cov_even_per_band.append(cov_even_save_per_band)


                                                #tab_targets.band1.append(nn[0])
                                                #tab_targets.band2.append(nn[1])
                                                #tab_targets.band3.append(nn[2])   
                                            try:
                                                tab_targets.des_id.append(params_image_sims[p0]['des_id'])
                                                tab_targets.photoz.append(params_image_sims[p0]['photoz'])
                                            except:
                                                pass

                                    elif im_type == 'image_m':
                                        if (mom.even ==  mom.even)[0]:
                                            tab_targets_m.add(mom, xy=newcent,id=ix,covgal=MomentCovariance(covgal[0],covgal[1]))

                                            tab_targets_m.p0.append(p0)
                                            tab_targets_m.p0_PSF.append(p0_PSF)
                                            tab_targets_m.ra.append(newcent[0])
                                            tab_targets_m.dec.append(newcent[1])
                                            tab_targets_m.ra_shift.append(newcent_shift[0])
                                            tab_targets_m.dec_shift.append(newcent_shift[1])
                                            tab_targets_m.bkg.append(bkg)
                                            
                                            tab_targets_m.len_v.append(len_v)

                                            #meb_ = np.array([m_.even for m_ in meb])
                                            tab_targets_m.meb.append(meb[0,:])
                                            tab_targets_m.AREA.append(0.)
                                            try:
                                                tab_targets_m.true_fluxes.append(fluxes)
                                            except:
                                                pass
                                            try:
                                                tab_targets_m.is_it_a_blend.append(count_blends)
                                            except:
                                                pass
                                            Mf,Mr,M1,M2,_ = mul_PSF.get_moment(0.,0.).even
                                    #
                                            tab_targets_m.psf_Mf.append(Mf)
                                            tab_targets_m.psf_Mr.append(Mr)
                                            tab_targets_m.psf_M1.append(M1)
                                            tab_targets_m.psf_M2.append(M2)

                                            tab_targets_m.cov_odd_per_band.append(cov_odd_save_per_band)
                                            tab_targets_m.cov_even_per_band.append(cov_even_save_per_band)


                                                #tab_targets.band1.append(nn[0])
                                                #tab_targets.band2.append(nn[1])
                                                #tab_targets.band3.append(nn[2])   
                                            try:
                                                tab_targets_m.des_id.append(params_image_sims[p0]['des_id'])
                                                tab_targets_m.photoz.append(params_image_sims[p0]['photoz'])
                                            except:
                                                pass


                # measure the additional stamp for selection purposes [applies only if RANDOM + DETECTION + POISSON] **********************************
                if ( (config['poisson']) & (config['mode_detection'] =='detection')):
                   
                    images_a = []
                    wcs_a = []
                    psf_a = []
                    bands_a = []
                    noise_a = []
                    for band in config['bands']:
                        maskx = (np.arange(config['size_tile'])>=x_a_poisson_selection-20) & (np.arange(config['size_tile'])<x_a_poisson_selection-20+40)
                        masky = (np.arange(config['size_tile'])>=y_a_poisson_selection-20) & (np.arange(config['size_tile'])<y_a_poisson_selection-20+40)
                        psf_p = Input_catalog[imc_keys[np.int(0)]][band]['psf_p']
                        gal_p = Input_catalog[imc_keys[np.int(0)]][band]['gal_p']

                        
                        
                        cent=(x_a_poisson_selection,y_a_poisson_selection)
                        origin = (0.,0.)
                        duv_dxy = np.array([[0.263, 0.],
                                            [0., 0.263]])
                        wcs_ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)
                    
                    
                        wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0])
                        
                        
                        
                        p0 = Input_catalog[imc_keys[np.int(0)]][band]['p0']
                        image_stamp = tile[band][im_type][masky,:][:,maskx]
                        if config['shredder']:
                            image_stamp = image_stamp-(shredder_cat[im_type]['models'][0])[masky,:][:,maskx]
                        if config['perfect_deblender']:
                            image_stamp = image_stamp-tile[b][im_type+'_noisefree'][masky,:][:,maskx]
                            
                        image_stamp += config['background']    
                  
                        if config['background_subtraction'] == True:
                            bkg,len_v = subtract_background_(image_stamp, np.zeros_like(image_stamp), np.zeros_like(image_stamp))
                            #print (subtract_background_(image_stamp, seg_, mask_),subtract_background_(tile[band][im_type][masky,:][:,maskx], seg_, mask_))
                        else:
                            bkg = 0.  
                            len_v = 100000000000.
                       # bkg = config['background']  
                       # len_v = 100000000000.
                        
                        image_stamp -=bkg
                                
                                
                                
                                
                        wt_stamp = tile[band][im_type][masky,:][:,maskx]                       


                        _,  psf_image,jac = render_gal(gal_p,psf_p,wcs_,image_stamp.shape[0], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)

                        
                        images_a.append(image_stamp)
                        wcs_a.append(wcs_)
                        psf_a.append(psf_image)
                        bands_a.append(band)
                        noise_a.append(tile[band]['noise_level'])

                    kds = bfd.multiImage(images_a, (0,0), psf_a, wcs_a, pixel_noiselist = noise_a, bandlist = bands_a ,pad_factor= config['pad_factor'])
                    wt = mc.KSigmaWeight(sigma = config['sigma']) 
                    mul = bfd.MultiMomentCalculator(kds, wt, bandinfo = config['band_dict_code'])
                    
                    #xyshift, error,msg = mul.recenter()
                    moments = mul

                    mom,meb_ = moments.get_moment(0,0,returnbands=True)
                    
                    
       
                    meb = np.array([m_.even for m_ in meb_])
                            
                    covm_even,covm_odd,covm_even_all,covm_odd_all = moments.get_covariance(returnbands=True)


                    psfn_ = np.zeros_like(psf_a[0])
                    psfn_[len(psfn_[0])//2,len(psfn_[0])//2] = 1
                  
                              
                        
                    kds_PSF = bfd.multiImage(psf_a, (0,0), [psfn_]*len(psf_a), wcs_a, pixel_noiselist = noise_a, bandlist = bands_a,pad_factor=config['pad_factor'])
                    mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, bandinfo = config['band_dict_code'])
                    xyshift, error,msg = mul_PSF.recenter()
                    
                    newcent = [x_a_poisson_selection,y_a_poisson_selection]
                    newcent_shift = xyshift/0.263

                    covgal = covm_even,covm_odd
                    covgal_per_band = covm_even_all,covm_odd_all 
                    if covgal_per_band is not None:
                        
                        cov_even_save_per_band = []
                        cov_odd_save_per_band = []
                        for ii in range(covgal_per_band[0].shape[0]):
                            cov_even_save_per_band.extend(covgal_per_band[0][ii][ii:])
                        for ii in range(covgal_per_band[1].shape[0]):
                            cov_odd_save_per_band.extend(covgal_per_band[1][ii][ii:])

                    mom.even *= 0.
                    
                    
                    
                    if im_type == 'image_p':
                        tab_targets.add(mom, xy=newcent,id=ix,covgal=MomentCovariance(covgal[0],covgal[1]))
                        tab_targets.p0.append(p0)
                        tab_targets.p0_PSF.append(p0_PSF)
                        tab_targets.ra.append(newcent[0])
                        tab_targets.dec.append(newcent[1])
                        tab_targets.ra_shift.append(newcent_shift[0])
                        tab_targets.dec_shift.append(newcent_shift[1])


                        tab_targets.AREA.append(AREA)
                        tab_targets.bkg.append(bkg)
                        tab_targets.len_v.append(len_v)
                        
                        
                        #print ('ip, ' , AREA)

                        tab_targets.meb.append(meb[0,:]*0.)
                        try:
                            tab_targets.true_fluxes.append(fluxes)
                        except:
                            pass
                        try:
                            tab_targets_m.is_it_a_blend.append(0)
                        except:
                            pass
                        Mf,Mr,M1,M2,_ = mul_PSF.get_moment(0.,0.).even
                #
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
                    
                    if im_type == 'image_m':
                    
                    
                        tab_targets_m.add(mom, xy=newcent,id=ix,covgal=MomentCovariance(covgal[0],covgal[1]))
                        tab_targets_m.p0.append(p0)
                        tab_targets_m.p0_PSF.append(p0_PSF)
                        tab_targets_m.ra.append(newcent[0])
                        tab_targets_m.dec.append(newcent[1])
                        tab_targets_m.ra_shift.append(newcent_shift[0])
                        tab_targets_m.dec_shift.append(newcent_shift[1])
                        tab_targets_m.AREA.append(AREA)
                        tab_targets_m.bkg.append(bkg)
                        tab_targets_m.len_v.append(len_v)
                        
                        
                       # print ('im, ' , AREA)

                        tab_targets_m.meb.append(meb[0,:]*0.)
                        try:
                            tab_targets_m.true_fluxes.append(fluxes)
                        except:
                            pass
                        Mf,Mr,M1,M2,_ = mul_PSF.get_moment(0.,0.).even
                #
                        tab_targets_m.psf_Mf.append(Mf)
                        tab_targets_m.psf_Mr.append(Mr)
                        tab_targets_m.psf_M1.append(M1)
                        tab_targets_m.psf_M2.append(M2)
                        tab_targets_m.cov_odd_per_band.append(cov_odd_save_per_band)
                        tab_targets_m.cov_even_per_band.append(cov_even_save_per_band)

                        try:
                            tab_targets_m.des_id.append(params_image_sims[p0]['des_id'])
                            tab_targets_m.photoz.append(params_image_sims[p0]['photoz'])
                        except:
                            pass
                
                
                
                
    
    
    
    
    
        #********************************************************
        #                         SAVE                         #
        #********************************************************
    
        
        if (  (config['poisson']) & (config['mode_detection'] =='detection')):
            STAMP_SIM = False
        else:
            STAMP_SIM = True
        if not do_templates:      
            if STAMP_SIM:
                s = 1
            else:
                s = 0
            save_(tab_targets,config['output_folder']+'/targets/'+'ISp_targets_{0}.fits'.format(ii_chunk),s)
            save_(tab_targets_m,config['output_folder']+'/targets/'+'ISm_targets_{0}.fits'.format(ii_chunk),s)

            if config['debug']:
                extra_info = dict()
                extra_info['input_coordinates'] = [x_a,y_a,x_a_p,y_a_p,x_a_m,y_a_m]
                try:
                    extra_info['detected_coordinates'] = [detection_cat[config['bands'][0]]['image_p']['x'],detection_cat[config['bands'][0]]['image_p']['y'],detection_cat[config['bands'][0]]['image_m']['x'],detection_cat[config['bands'][0]]['image_m']['y']]
                    extra_info['detected_coordinates_sel'] = [final_coordinates['image_p'],final_coordinates['image_m']]
                except:
                    pass
                
                extra_info['noiseless tile'] = [tile[b]['image_p']-tile[b]['noise']]
                extra_info['tile'] = [tile[b]['image_p']]
                extra_info['seg_map'] = [seg_map[b]['image_p']]
                
                
                np.save(config['output_folder']+'/targets/debug_images{0}'.format(ii_chunk),[debug_images,extra_info])
            
            
            
        else:

            path = config['output_folder']+'/templates/'+'/IS_templates__chunk_'+str(ii_chunk)

            if config['debug']:
                '''
                need to save:
                - input coordinates
                - sheared input coordinates
                - detection list
                - detection list after re-grouping things together
                
                '''
                
                extra_info = dict()
                extra_info['input_coordinates'] = [x_a,y_a]
             
                try:
                    extra_info['detected_coordinates'] = [detection_cat[config['bands'][0]]['image_n']['x'],detection_cat[config['bands'][0]]['image_n']['y']]
                    extra_info['detected_coordinates_sel'] = [final_coordinates['image_n']]
                except:
                    pass
                extra_info['noiseless tile'] = [tile[b]['image_n']-tile[b]['noise']]
                extra_info['tile'] = [tile[b]['image_n']]
                np.save(config['output_folder']+'/templates/debug_images{0}'.format(ii_chunk),[debug_images,extra_info])

                
            tab_detections.EFFAREA = len(np.arange(len_loop)[indexes_final])
            # we need to save the AreA of the tiles. such thtat the final number density will be the sum of all the areas
            tab_detections.AREA_tile = AREA
            
            
            
        
        
            save__ = dict()
            count = 0
            for index in range(len(tab_detections.images)):

                cc = False
                try:
                #if 1==1:
                    mom = tab_detections.images[index].moments.get_moment(0.,0.)
                    if (mom.even ==  mom.even)[0]:
                        cc = True
                except:
                #    print ('not a valid measurement')
                    pass
                if cc:

                    count +=1
                    save__[index] = dict()
                    save__[index]['moments'] = tab_detections.images[index].moments
                    save__[index]['index_gal'] = int(ii_chunk*10000)+index#tab_detections.images[index].image_ID[0]
                    save__[index]['index'] = tab_detections.images[index].image_ID[0]
                    try:
                        save__[index]['MOF_index'] = tab_detections.images[index].MOF_index
                        save__[index]['MAG_I'] = tab_detections.images[index].MAG_I
                        save__[index]['tilename'] = tab_detections.images[index].TILENAME                
                    except:
                        pass
                    try:
                        save__[index]['ra'] = tab_detections.images[index].image_ra[0]
                        save__[index]['dec'] = tab_detections.images[index].image_dec[0]
                    except:
                        pass
                    save__[index]['p0']= tab_detections.images[index].p0
                    save__[index]['p0_PSF']= tab_detections.images[index].p0_PSF
                    try:
                        save__[index]['bkg'] = tab_detections.images[index].bkg
                        save__[index]['len_v'] = tab_detections.images[index].len_v
                        
                    except:
                        pass

            save_obj(path,save__)
            print ('done')

  
            
            
            path_A = config['output_folder']+'/templates/'+'/AIS_templates__chunk_'+str(ii_chunk)
            #print ('')
            #print (len(np.arange(len_loop)[indexes_final]),STAMP_SIM)
            save_obj(path_A,[len(np.arange(len_loop)[indexes_final]),AREA,STAMP_SIM])
            
            



def make_tiles_tt(output_folder,**config):
    print ('doing something')
    config['output_folder'] = output_folder
    
    if config['MPI']:
        from mpi4py import MPI  
    
    if not os.path.exists(output_folder+'/targets/'):
        try:
            os.mkdir(output_folder+'/targets/')
        except:
                pass
    if not os.path.exists(output_folder+'/templates/'):
        try:
            os.mkdir(output_folder+'/templates/')
        except:
                pass
            
    number_of_runs_targets = math.ceil(config['n_targets']/config['gal_per_tile'])
    number_of_runs_templates = math.ceil(config['n_templates']/config['gal_per_tile'])

    params_image_sims = np.load(config['models'] ,allow_pickle=True).item()
    
  
    #mute = dict()
    #for i in range(len(config['band_dict'])):
    #    mute[config['band_dict'][i][0]] = BandInfo(config['band_dict'][i][1],i)
    #    
    uu__ = dict()
    mute_b = dict()
    for i in range(len(config['band_dict'])):
        mute_b[config['band_dict'][i][0]] = config['band_dict'][i][1]
    #params_template['band_dict'] = mute
    uu__['bands'] = config['bands']
    uu__['band_dict'] = dict()
    uu__['band_dict']['bands'] = list(config['bands'])
    w = []
    for b in config['bands']:
        w.append(mute_b[b])
    uu__['band_dict']['weights'] = list(w)
    uu__['band_dict']['index'] = list(np.arange(len(w)))


            
    config['band_dict_code'] = uu__['band_dict']
    print ('number_of_runs_targets: ',number_of_runs_targets)
    print ('number_of_runs_templates: ',number_of_runs_templates)
    
    if 'targets' in config['do']:
        run_count = 0
        
        if config['MPI']:
            while run_count<number_of_runs_targets:
                comm = MPI.COMM_WORLD
                if (run_count+comm.rank) < number_of_runs_targets:
                    pipeline_targets(config,params_image_sims, run_count+comm.rank)
                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        else:
       
            while run_count<number_of_runs_targets:

                if (run_count) < number_of_runs_targets:
                    pipeline_targets(config,params_image_sims, run_count)
                run_count+=1
        
    
    if 'templates' in config['do']:
        run_count = 0
        if config['MPI']:
            while run_count<number_of_runs_templates:
                comm = MPI.COMM_WORLD
                if (run_count+comm.rank) < number_of_runs_templates:
                    pipeline_targets(config,params_image_sims, run_count+comm.rank,do_templates=True)
                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        else:
            while run_count<number_of_runs_templates:

                if (run_count) < number_of_runs_templates:
                    pipeline_targets(config,params_image_sims, run_count,do_templates=True)
                run_count+=1

            


        
