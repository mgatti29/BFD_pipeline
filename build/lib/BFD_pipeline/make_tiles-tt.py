from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
from ngmix.jacobian import Jacobian
import os, sys
from .read_meds_utils import Image, MOF_table, DetectionsTable,render_gal
import copy
from astropy import units as uu
from astropy.coordinates import SkyCoord
from bfd import momentcalc as mc
import bfd
import sxdes

'''
This code simulate a tile, targets & templates *****

'''


import pytest
import sxdes


FWHM_FAC = 2*np.sqrt(2*np.log(2))
config_meds = dict()
config_meds['max_box_size'] = 48
config_meds['min_box_size'] = 32
config_meds["allowed_box_sizes"] = [
                2, 3, 4, 6, 8, 12, 16, 24, 32, 48,
                64, 96, 128, 192, 256,
                384, 512, 768, 1024, 1536,
                2048, 3072, 4096, 6144 ]

config_meds["sigma_fac"] = 5.0
config_meds["refband"] = "r"
config_meds["sub_bkg"] = False
config_meds["magzp_ref"] = 30.
config_meds["stage_output"] = False
config_meds["use_rejectlist"] = True
            
# Choose the boxsize - this is the same method as used in desmeds
# Pasted in these functions from desmeds.
def get_box_sizes(cat):
    """
    get box sizes that are wither 2**N or 3*2**N, within
    the limits set by the user
    """
    sigma_size = get_sigma_size(cat)

    # now do row and col sizes
    row_size = cat['ymax'] - cat['ymin'] + 1
    col_size = cat['xmax'] - cat['xmin'] + 1

    # get max of all three
    box_size = np.vstack(
        (col_size, row_size, sigma_size)).max(axis=0)

    # clip to range
    box_size = box_size.clip(
        config_meds['min_box_size'], config_meds['max_box_size'])

    # now put in fft sizes
    bins = [0]
    bins.extend([sze for sze in config_meds['allowed_box_sizes']
                 if sze >= config_meds['min_box_size']
                 and sze <= config_meds['max_box_size']])

    if bins[-1] != config_meds['max_box_size']:
        bins.append(config_meds['max_box_size'])

    bin_inds = np.digitize(box_size, bins, right=True)
    bins = np.array(bins)

    return bins[bin_inds]

def get_sigma_size(cat):
    """
    "sigma" size, based on flux radius and ellipticity
    """
    ellipticity = 1.0 #- cat['b_world']/cat['a_world']
    sigma = cat['flux_radius']*2.0/FWHM_FAC
    drad = sigma*config_meds['sigma_fac']
    drad = drad*(1.0 + ellipticity)
    drad = np.ceil(drad)
    # sigma size is twice the radius
    sigma_size = 2*drad.astype('i4')

    return sigma_size


    
def _make_image(rng):
    dims = 32, 32
    sigma = 2.0
    counts = 100.0
    noise = 0.1

    cen = (np.array(dims)-1)/2
    cen += rng.uniform(low=-2, high=2, size=2)

    rows, cols = np.mgrid[
        0: dims[0],
        0: dims[1],
    ]

    rows = rows - cen[0]
    cols = cols - cen[1]

    norm = 1.0/(2 * np.pi * sigma**2)

    image = np.exp(-0.5*(rows**2 + cols**2)/sigma**2)
    print (counts*norm)
    image *= counts*norm

    # import hickory
    # plt = hickory.Plot(aratio=1)
    # plt.imshow(image)
    # plt.show()
    return image, noise, cen


def test_detect_smoke():
    rng = np.random.RandomState(646509750)
    image, noise, _ = _make_image(rng)
    _ = sxdes.run_sep(image, noise)


def test_detect():
    rng = np.random.RandomState(60970)

    for i in range(10):
        image, noise, cen = _make_image(rng)
        cat, seg = sxdes.run_sep(image, noise)
        assert cat.size > 0

        s = cat['flux'].argsort()[-1]
        row, col = cat['y'][s], cat['x'][s]
        assert abs(row - cen[0]) < 1
        assert abs(col - cen[1]) < 1


def test_mask():
    rng = np.random.RandomState(60970)

    for i in range(10):
        image, noise, cen = _make_image(rng)
        mask = np.ones(image.shape, dtype=bool)
        cat, seg = sxdes.run_sep(image, noise, mask=mask)
        assert cat.size == 0


def test_errors():
    rng = np.random.RandomState(60970)

    image, noise, cen = _make_image(rng)
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        cat, seg = sxdes.run_sep(image, noise, mask=mask)


def test_seg():
    rng = np.random.RandomState(60970)

    for i in range(10):
        image, noise, cen = _make_image(rng)
        cat, seg = sxdes.run_sep(image, noise)
        assert cat.size > 0

        for i in range(1, cat.size+1):
            msk = seg == i
            assert cat["isoarea_image"][i-1] == np.sum(msk)






def pipeline_targets(config, params_image_sims, i, do_templates = False):
    if do_templates:
        path = output_folder+'/templates/'+'/templates__chunk_'+str(i)+'.pkl'
    else:
        path = config['output_folder']+'/targets/'+'ISm_chunk_{0}.fits'.format(i)
    if not os.path.exists(path):

        # create target file ********************************************************
        tab_targets = TargetTable(n = config['n'],sigma = config['sigma'], cov=None)

        tab_targets.ra = []
        tab_targets.dec = []
        tab_targets.psf_Mf = []
        tab_targets.psf_Mr = []
        tab_targets.psf_M1= []
        tab_targets.psf_M2= []
        tab_targets.band1 = []
        tab_targets.band2 = []
        tab_targets.band3 = []
        tab_targets.p0 = []
        tab_targets.p0_PSF = []
        tab_targets.meb = []
        tab_targets.true_fluxes = []
        tab_targets.cov_odd_per_band = []
        tab_targets.cov_even_per_band = []

        tab_targets.des_id = []
        tab_targets.photoz = []
        tab_targets_m = copy.deepcopy(tab_targets)

        if do_templates:
            tab = TemplateTable(n = config['n'],
                       sigma = config['sigma'],
                        sn_min = 0.,#self.params['sn_min'], 
                        sigma_xy = 0.,#self.params['sigma_xy'], 
                        sigma_flux = 0.,#self.params['sigma_flux'], 
                        sigma_step = 0.,#self.params['sigma_step'], 
                        sigma_max = 0.,#self.params['sigma_max'],
                        xy_max = 0.)#self.params['xy_max'])

            tab_detections = DetectionsTable(params_template)
        

           

        
        # set the noise ************************************************************
        if len(config['noise_ext'])==2:
            noise_ext = config['noise_ext'][0] + np.random.randint(0,10000,1)[0]*0.0001*(config['noise_ext'][1]-config['noise_ext'][0])
        else:
            noise_ext = config['noise_ext']
        
        tile = dict()
        # make the tile ***********************************************************
        for band in config['bands']:
            tile[band] = dict()
            tile[band]['image_p'] = np.zeros((config['size_tile'],config['size_tile']))
            tile[band]['image_m'] = np.zeros((config['size_tile'],config['size_tile']))
            #tile[band]['image_p_noisefree'] = np.zeros((config['size_tile'],config['size_tile']))
            #tile[band]['image_m_noisefree'] = np.zeros((config['size_tile'],config['size_tile']))

        # define the input positions for the galaxies. ****************************
        Input_catalog = dict()

        if config['grid']:
            spacing = np.int((config['size_tile']-80)/np.sqrt(config['gal_per_tile']))
            x_a = []
            y_a = []
            for i in range(np.int((config['size_tile']-80)/spacing)):
                for j in range(np.int((config['size_tile']-80)/spacing)):
                    x_a.append(40+np.int(spacing*0.5)+spacing*i)
                    y_a.append(40+np.int(spacing*0.5)+spacing*j)
        else:
            x_a = np.random.randint(40,config['size_tile']-40,config['gal_per_tile'])
            y_a = np.random.randint(40,config['size_tile']-40,config['gal_per_tile'])

        x_a = np.array(x_a)
        y_a = np.array(y_a)
        
        # generate galaxies at positions ********************************************
        for real in range(len(x_a)):

            # make WCS for the object ***********************
            cent=(y_a[real],x_a[real])
            origin = (0.,0.)
            duv_dxy = np.array([[0.263, 0.],
                                [0., 0.263]])
            wcs_ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)
            
            # choose galaxy & PSF model *****************
            pp = list(params_image_sims.keys())
            redoit = True
            count_repeat = 0 
            while redoit:
                if config['index_P0'] and count_repeat<20:
                    p0 = pp[config['index_P0']]
                    if do_templates:
                        p0 = pp[config['gal_per_tile']*i+real]
                else:
                    p0 = pp[np.random.randint(0,len(pp),1)[0]]
                if config['index_P0_PSF'] and count_repeat<200:
                    p0_PSF = pp[config['index_P0_PSF']]#pp[np.random.randint(0,len(pp),1)[0]]
                else:
                    p0_PSF = copy.copy(p0)
                redoit = True
                for b, band in enumerate(config['bands']):
                    gp = params_image_sims[p0][band]['gal_pars']

                    if (gp[2]**2+gp[3]**2)*config['resize_sn']**2>=0.95:
                        if b == 0:
                            redoit= True
                        else:
                            redoit = redoit | True
                    else:
                        if b == 0:
                            redoit= False
                        else:
                            redoit = redoit | False
                count_repeat += 1



            # read the galaxy and PSF and randomly rotate the galaxy

            twotheta = 2.0 * np.radians(np.random.randint(0,36000)*1./100.)
            cos2angle = np.cos(twotheta)
            sin2angle = np.sin(twotheta)


            if config['resize_sn'] != 1. :
                gp[3] *= config['resize_sn']
                gp[2] *= config['resize_sn']

                
            galaxy_info = dict()

            for b in config['bands']:
                gp = copy.deepcopy(params_image_sims[p0][b]['gal_pars'])
                psfp = params_image_sims[p0_PSF][b]['pfs_params']

                mm1= gp[2] * cos2angle +  gp[3] * sin2angle
                mm2= -gp[2] * sin2angle + gp[3] * cos2angle
                gp[2] = copy.copy(mm1)
                gp[3] = copy.copy(mm2)

                galaxy_info[b] = {'gal_p':gp,'psf_p':psfp,'wcs':wcs_}

                mute_p,simulated_psf,jac = render_gal(gp,psfp,wcs_,config['size_tile'], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)
                mute_m,jac = render_gal(gp,psfp,wcs_,config['size_tile'],  g1 = config['g1'][1], g2 = config['g2'][1])
                if (mute_p is not None) and (mute_m is not None):
                    tile[b]['image_p'] += mute_p
                    tile[b]['image_m'] += mute_m
                    #tile[b]['image_p_noisefree'] += mute_p
                    #tile[b]['image_m_noisefree'] += mute_m
                    Input_catalog[y_a[real],x_a[real]] = galaxy_info    
            # render noise ******************************************
            for b in config['bands']:
                if noise_ext:
                    noise = np.random.normal(size = (config['size_tile'],config['size_tile']))*noise_ext
                    tile[b]['noise_level']  = noise_ext*config['noise_factor']

                else:
                    print ('NOT SUPPORTED')
                    sys.exit()
                    try:
                        noise = np.random.normal(size = (config['size_tile'],config['size_tile']))*np.sqrt(1.0/tile[band]['weight'])
                    except:
                        noise = np.random.normal(size = (config['size_tile'],config['size_tile']))*np.sqrt(1.0/np.median(tile[band]['weight']))  
                tile[b]['noise'] = noise*config['noise_factor']


            if not config['noiseless']:
                tile[b]['image_p']+=noise*config['noise_factor']
                tile[b]['image_m']+=noise*config['noise_factor']

                        # apply mask
            if not config['maskless']:
                mask = ( tile[band]['weight']==0. )| (tile[band]['mask']!=0.)        
                tile[b]['image_p'][mask] = 0. 
                tile[b]['image_m'][mask] = 0. 

                tile[b]['image_p_noisefree'][mask] = 0. 
                tile[b]['image_m_noisefree'][mask] = 0. 

        # detection step ***************************
        
        detection_cat = dict()

        for b in config['bands']:
            detection_cat[b] = dict()
            detection_cat[b]['image_p'], seg_p = sxdes.run_sep(tile[band]['image_p'], config['noise_factor']*noise_ext)
            detection_cat[b]['image_m'], seg_m = sxdes.run_sep(tile[band]['image_m'], config['noise_factor']*noise_ext)

            
        '''
        extra_obj_data_fields = [('number', 'i8'),]
        obj_data = meds.util.get_meds_input_struct(len(cat['x']), extra_fields=extra_obj_data_fields)
        obj_data["id"] = cat['number']
        obj_data["number"] = cat['number']
        obj_data["ra"] = cat['x']
        obj_data["dec"] = cat['y']
        obj_data["box_size"] = get_box_sizes(cat)
        '''
        
        

        # measure moments **********************
        if config['mode_detection'] == 'input':
            for ix in range(len(Input_catalog.keys())):

                ll = np.array(list(Input_catalog.keys()))

                for im_type in ['image_p','image_m']:
                    images_a = []
                    wcs_a = []
                    psf_a = []
                    bands_a = []
                    noise_a = []
                    for band in config['bands']:
                        # cut the image **************************
                        maskx = (np.arange(config['size_tile'])>=ll[ix][0]-20) & (np.arange(config['size_tile'])<ll[ix][0]-20+40)
                        masky = (np.arange(config['size_tile'])>=ll[ix][1]-20) & (np.arange(config['size_tile'])<ll[ix][1]-20+40) 

                        image_stamp = tile[band][im_type][masky,:][:,maskx]
                        wt_stamp = tile[band][im_type][masky,:][:,maskx]
                       

                        psf_p = Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['psf_p']
                        gal_p = Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['gal_p']
                        wcs_ = copy.copy(Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['wcs'])
                        wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0])

                        # render galaxy and psf model *****
                        _,  psf_image,jac = render_gal(gal_p,psf_p,wcs_,image_stamp.shape[0], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)



                        images_a.append(image_stamp)
                        wcs_a.append(wcs_)
                        psf_a.append(psf_image)
                        bands_a.append(band)
                        noise_a.append(tile[band]['noise_level'])

                    kds = bfd.multiImage(images_a, (0,0), psf_a, wcs_a, pixel_noiselist = noise_a, bandlist = bands_a ,pad_factor= config['pad_factor'])
                    wt = mc.KSigmaWeight(sigma = config['sigma']) 
                    mul = bfd.MultiMomentCalculator(kds, wt, band_dict = band_dict)
                    xyshift, error,msg = mul.recenter()
                    moments = mul

                    mom,meb, mob = moments.get_moment(0,0,return_bands=True)
                    covm_even,covm_odd,covm_even_all,covm_odd_all = moments.get_covariance(return_bands=True)


                    kds_PSF = bfd.multiImage(psf_a, (0,0), [None]*len(psf_a), wcs_a, pixel_noiselist = noise_a, bandlist = bands_a,pad_factor=config['pad_factor'])
                    mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, band_dict = band_dict)

                    newcent=np.array([ll[np.int(ix)][0],ll[np.int(ix)][1]])+xyshift


                    covgal = covm_even,covm_odd
                    covgal_per_band = covm_even_all,covm_odd_all 
                    if covgal_per_band is not None:

                        cov_even_save_per_band = []
                        cov_odd_save_per_band = []
                        for ii in range(covgal_per_band[0].shape[0]):
                            cov_even_save_per_band.extend(covgal_per_band[0][ii][ii:])
                        for ii in range(covgal_per_band[1].shape[0]):
                            cov_odd_save_per_band.extend(covgal_per_band[1][ii][ii:])

                    if im_type == 'image_p':
                        tab_targets.add(mom, xy=newcent,id=ix,number=1,covgal=covgal)
                        tab_targets.p0.append(p0)
                        tab_targets.p0_PSF.append(p0_PSF)
                        tab_targets.ra.append(newcent[0])
                        tab_targets.dec.append(newcent[1])

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

                    elif im_type == 'image_p':
                        tab_targets_m.add(mom, xy=newcent,id=ix,number=1,covgal=covgal)
                        tab_targets_m.p0.append(p0)
                        tab_targets_m.p0_PSF.append(p0_PSF)
                        tab_targets_m.ra.append(newcent[0])
                        tab_targets_m.dec.append(newcent[1])

                        tab_targets_m.meb.append(meb[0,:])
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


                            #tab_targets.band1.append(nn[0])
                            #tab_targets.band2.append(nn[1])
                            #tab_targets.band3.append(nn[2])   
                        try:
                            tab_targets_m.des_id.append(params_image_sims[p0]['des_id'])
                            tab_targets_m.photoz.append(params_image_sims[p0]['photoz'])
                        except:
                            pass
    
    
        if config['mode_detection'] == 'detection':
            for im_type in ['image_p','image_m'] :
                extra_obj_data_fields = [('number', 'i8'),]
                obj_data = meds.util.get_meds_input_struct(len(detection_cat[config['bands'][0]][im_type]['x']), extra_fields=extra_obj_data_fields)
                obj_data["id"] = detection_cat[config['bands'][0]][im_type]['number']
                obj_data["number"] = detection_cat[config['bands'][0]][im_type]['number']
                obj_data["ra"] = detection_cat[config['bands'][0]][im_type]['x']
                obj_data["dec"] = detection_cat[config['bands'][0]][im_type]['y']
                obj_data["box_size"] = get_box_sizes(detection_cat[config['bands'][0]][im_type])
                for ix in range(len(detection_cat[config['bands'][0]][im_type]['y'])):
                    
                    # determin the size stamp
                    minx = np.int(detection_cat[config['bands'][0]][im_type]['x'][ix]-obj_data["box_size"][ix]/2)
                    miny = np.int(detection_cat[config['bands'][0]][im_type]['y'][ix]-obj_data["box_size"][ix]/2)

                    box_size = obj_data["box_size"][ix]
                    maskx = (np.arange(config['size_tile'])>=detection_cat[config['bands'][0]][im_type]['x'][ix]-box_size/2) & (np.arange(config['size_tile'])<detection_cat[config['bands'][0]][im_type]['x'][ix]-box_size/2+box_size)
                    masky = (np.arange(config['size_tile'])>=detection_cat[config['bands'][0]][im_type]['y'][ix]-box_size/2) & (np.arange(config['size_tile'])<detection_cat[config['bands'][0]][im_type]['y'][ix]-box_size/2+box_size) 

    
                    #match to input catalog to find PSF & noise model for the stamp *******
                    ll = np.array(list(Input_catalog.keys()))
                  


                    goldcat = SkyCoord(ra=[detection_cat[config['bands'][0]][im_type]['x'][ix]*uu.degree*0.263/60.], dec=[detection_cat[config['bands'][0]][im_type]['y'][ix]*uu.degree*0.263/60.])  
                    catalog = SkyCoord(ra=ll[:,0]*uu.degree*0.263/60., dec=ll[:,1]*uu.degree*0.263/60.)  
                    idx, d2d, d3d = goldcat.match_to_catalog_sky(catalog, nthneighbor=1) 
                    dist_pix = np.sqrt((detection_cat[config['bands'][0]][im_type]['x'][ix]-ll[idx][0][0])**2+(detection_cat[config['bands'][0]][im_type]['y'][ix]-ll[idx][0][1])**2)

        
                    images_a = []
                    wcs_a = []
                    psf_a = []
                    bands_a = []
                    noise_a = []

                    for band in config['bands']:
                        # cut the image **************************
                        maskx = (np.arange(config['size_tile'])>=ll[ix][0]-20) & (np.arange(size_tile)<ll[ix][0]-20+40)
                        masky = (np.arange(config['size_tile'])>=ll[ix][1]-20) & (np.arange(config['size_tile'])<ll[ix][1]-20+40) 

                        image_stamp = tile[band][im_type][masky,:][:,maskx]
                        wt_stamp = tile[band][im_type][masky,:][:,maskx]
              

                        psf_p = Input_catalog[(ll[idx][0],ll[idx][1])][band]['psf_p']
                        gal_p = Input_catalog[(ll[idx][0],ll[idx][1])][band]['gal_p']
                        wcs_ = copy.copy(Input_catalog[(ll[idx][0],ll0[idx][1])][band]['wcs'])
                        wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0])

                        # render galaxy and psf model *****
                        _,  psf_image,jac = render_gal(gal_p,psf_p,wcs_,image_stamp.shape[0], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)



                        images_a.append(image_stamp)
                        wcs_a.append(wcs_)
                        psf_a.append(psf_image)
                        bands_a.append(band)
                        noise_a.append(tile[band]['noise_level'])

                    kds = bfd.multiImage(images_a, (0,0), psf_a, wcs_a, pixel_noiselist = noise_a, bandlist = bands_a ,pad_factor= config['pad_factor'])
                    wt = mc.KSigmaWeight(sigma = config['sigma']) 
                    mul = bfd.MultiMomentCalculator(kds, wt, band_dict = band_dict)
                    xyshift, error,msg = mul.recenter()
                    moments = mul

                    mom,meb, mob = moments.get_moment(0,0,return_bands=True)
                    covm_even,covm_odd,covm_even_all,covm_odd_all = moments.get_covariance(return_bands=True)


                    kds_PSF = bfd.multiImage(psf_a, (0,0), [None]*len(images_a), wcs_a, pixel_noiselist = noise_a, bandlist = bands_a,pad_factor=config['pad_factor'])
                    mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, band_dict = band_dict)

                    newcent=np.array([ll[np.int(ix)][0],ll[np.int(ix)][1]])+xyshift


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
                        Wide_g = Image(index, meds = [], bands = config['bands'])
                        tab_detections.add_image(Wide_g)
                        tab_detections[index].image.moments = mul
            
            
                    if not do_templates:
                        if im_type == 'image_p':
                            tab_targets.add(mom, xy=newcent,id=ix,number=1,covgal=covgal)
                            tab_targets.p0.append(p0)
                            tab_targets.p0_PSF.append(p0_PSF)
                            tab_targets.ra.append(newcent[0])
                            tab_targets.dec.append(newcent[1])

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
                            tab_targets_m.add(mom, xy=newcent,id=ix,number=1,covgal=covgal)
                            tab_targets_m.p0.append(p0)
                            tab_targets_m.p0_PSF.append(p0_PSF)
                            tab_targets_m.ra.append(newcent[0])
                            tab_targets_m.dec.append(newcent[1])

                            tab_targets_m.meb.append(meb[0,:])
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


                                #tab_targets.band1.append(nn[0])
                                #tab_targets.band2.append(nn[1])
                                #tab_targets.band3.append(nn[2])   
                            try:
                                tab_targets_m.des_id.append(params_image_sims[p0]['des_id'])
                                tab_targets_m.photoz.append(params_image_sims[p0]['photoz'])
                            except:
                                pass

        if not do_templates:            
            save_(tab_targets,config['output_folder']+'/targets/'+'ISp_chunk_{0}.fits'.format(i))
            save_(tab_targets_m,config['output_folder']+'/targets/'+'ISm_chunk_{0}.fits'.format(i))

        else:
            path = output_folder+'/templates/'+'/templates__chunk_'+str(i)
            
            tab_detections.EFFAREA = len(x_a)
            save_obj(path,tab_detections)





from mpi4py import MPI  

def make_tiles_tt(output_folder,**config):
    
    config['output_folder'] = output_folder
    
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
    run_count = 0
    while run_count<number_of_runs_targets:
        comm = MPI.COMM_WORLD
        if (run_count+comm.rank) < number_of_runs_targets:
            pipeline_targets(config,params_image_sims, run_count+comm.rank)
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
    run_count = 0
    while run_count<number_of_runs_templates:
        comm = MPI.COMM_WORLD
        if (run_count+comm.rank) < number_of_runs_templates:
            pipeline_templates(config,params_image_sims, run_count+comm.rank,do_templates=True)
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
