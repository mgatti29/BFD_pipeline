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
import numpy as np
import math
from .utilities import update_progress, save_obj, load_obj
import astropy.io.fits as fits
import frogress
import ngmix
import meds
'''
This code simulate a tile, targets & templates *****

'''


import pytest
import sxdes
from bfd.momentcalc import MomentCovariance

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


def save_(self,fitsname,stamp):
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
        #col.append(fits.Column(name="number",format="K",array=self.id))
        
        col.append(fits.Column(name="ra",format="D",array=self.ra))
        col.append(fits.Column(name="dec",format="D",array=self.dec))
        col.append(fits.Column(name="ra_shift",format="D",array=self.ra_shift))
        col.append(fits.Column(name="dec_shift",format="D",array=self.dec_shift))
        
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
        if self.cov is not None:
            col.append(fits.Column(name="covariance",format="15E",array=np.array(self.cov).astype(np.float32)))
        
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
       #    
            
            
        #if self.cov is not None:
        #    col.append(fits.Column(name="covariance",format="15E",array=self.cov.astype(np.float32)))
        #if self.area is not None:
        #    col.append(fits.Column(name="area",format="E",array=self.cov.astype(np.float32)))
        #if self.select is not None:
        #    col.append(fits.Column(name="select",format="I",array=self.cov.astype(np.int16)))
        #    
            
        
        col.append(fits.Column(name="area",format="D",array=self.AREA))
        
        cols=fits.ColDefs(col)
            
        #print (len(self.id))
        #print (len(self.p0))
        #print (len(self.p0_PSF))
        #print (np.array(self.meb).shape)
        #print (np.array(self.true_fluxes).shape)
        #print (np.array(self.moment).shape)
        #print (len(self.xy))
        #print (len(self.ra))
        #print (len(self.ra_shift))
        #print (len(self.psf_Mf))
        #print (np.array(self.cov_even_per_band).shape)
        #print (len(self.ra_shift))

        tbhdu = fits.BinTableHDU.from_columns(cols)
        #self.prihdu.header['NLOST'] = self.nlost  # Update value
        self.prihdu.header['STAMP'] = stamp
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
      
        return
    
    
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



# TODO: when there's a bad detection and we get a NaN, we need to re-measure without the recentering!


def pipeline_targets(config, params_image_sims, ii_chunk, do_templates = False):
    print ('running chunk -'+str(ii_chunk))

    if do_templates:
        path = config['output_folder']+'/templates/'+'/IS_templates__chunk_'+str(ii_chunk)+'.pkl'
    else:
        path = config['output_folder']+'/targets/'+'ISm_targets_{0}.fits'.format(ii_chunk)
    if not os.path.exists(path):

        # create target file ********************************************************
        tab_targets = TargetTable(n = config['n'],sigma = config['sigma'])

        tab_targets.ra = []
        tab_targets.dec = []
        tab_targets.AREA = []
        tab_targets.ra_shift = []
        tab_targets.dec_shift = []
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
            params_template = {}
            params_template['n'] = config['n'] # 4 default. KSigmaWeight function index   
            params_template['sigma'] = config['sigma'] # sigma KSigmaWeight         
            mute = dict()
            for i in range(len(config['band_dict'])):
                mute[config['band_dict'][i][0]] = BandInfo(config['band_dict'][i][1],i)
            params_template['band_dict'] = mute
            params_template['bands'] = config['bands']
            tab = TemplateTable(n = config['n'],
                       sigma = config['sigma'],
                        sn_min = 0.,#self.params['sn_min'], 
                        sigma_xy = 0.,#self.params['sigma_xy'], 
                        sigma_flux = 0.,#self.params['sigma_flux'], 
                        sigma_step = 0.,#self.params['sigma_step'], 
                        sigma_max = 0.,#self.params['sigma_max'],
                        xy_max = 0.)#self.params['xy_max'])

            tab_detections = DetectionsTable(params_template)
        
        
        if do_templates:
            replicas = 1
        else:
            replicas = config['number_of_replicas']

        for rep in frogress.bar(range(replicas)):

        
            # set the noise ************************************************************
            if do_templates:
                if len(config['noise_ext_templates'])==2:
                    noise_ext = config['noise_ext_templates'][0] + np.random.randint(0,10000,1)[0]*0.0001*(config['noise_ext_templates'][1]-config['noise_ext_templates'][0])
                else:
                    noise_ext = config['noise_ext_templates'][0]
            else:
                if len(config['noise_ext'])==2:
                    noise_ext = config['noise_ext'][0] + np.random.randint(0,10000,1)[0]*0.0001*(config['noise_ext'][1]-config['noise_ext'][0])
                else:
                    noise_ext = config['noise_ext'][0]

            tile = dict()
            
            
            # make the tile ***********************************************************
            for band in config['bands']:
                tile[band] = dict()
                if do_templates:
                     tile[band]['image_n'] = np.zeros((config['size_tile'],config['size_tile']))
                else:
                    tile[band]['image_p'] = np.zeros((config['size_tile'],config['size_tile']))
                    tile[band]['image_m'] = np.zeros((config['size_tile'],config['size_tile']))

            # define the input positions for the galaxies. ****************************
            
            Input_catalog = dict()

            if do_templates:
                xxe = config['grid_templates']
            else:
                xxe = config['grid_targets']
            if xxe:
                spacing = math.ceil((config['size_tile']-80)/np.sqrt(config['gal_per_tile']))
                x_a = []
                y_a = []
                for i in range(math.ceil((config['size_tile']-80)/spacing)):
                    for j in range(math.ceil((config['size_tile']-80)/spacing)):
                        x_a.append(40+np.int(spacing*0.5)+spacing*i)
                        y_a.append(40+np.int(spacing*0.5)+spacing*j)
            else:
                if config['poisson']:
                    obj_ = np.random.poisson(config['gal_per_tile'])
                else:
                    obj_ =config['gal_per_tile']
                x_a = np.random.randint(40,config['size_tile']-40,obj_)
                y_a = np.random.randint(40,config['size_tile']-40,obj_)

            x_a = np.array(x_a)[:config['gal_per_tile']]
            y_a = np.array(y_a)[:config['gal_per_tile']]
            AREA =  (config['size_tile']-80)**2
            
            
            
            x_a_poisson_selection = np.random.randint(40,config['size_tile']-40,1)[0]
            y_a_poisson_selection = np.random.randint(40,config['size_tile']-40,1)[0]

            
            '''
            we have a STAMP sim in the following cases:
            (GRID  has no poisson option)
            - GRID INPUT  (1 if NaN)
            - GRID DETECTION (1 if NaN or missed INPUT)
            - RANDOM INPUT NO POIS. (1 if NaN)
            - RANDOM DETECTION NO POIS (1 if NaN or missed INPUT)
            we have a poisson detection sims if:
            - RANDOM DETECTION POIS. (1 adition stamp, area of the tile.)
            
            for templates,
            
            DENSITY = config['gal_per_tile']/AREA # -> this actually comes from the detected templates!!!
            
            STAMP sims:
                sky_density =  1./templates_tot (for templates)
                area of non detection: 1.

            goal: N_missed.

            POISSON SIMS sims:
                sky_density = gal_per_tile/area_tile * 1./ templates_tot ~ 1/ area_tot (deep fields) (for templates)
                area_for_pseudo_target = area_of_tile (if tile is homoegenous) # 1 per tile.

            goal: N_tot
            
            maybe random: 
            
            -> TODO: (STAMP): assign always area = 1, 0 moments, BUT the right covariance.
            -> TODO: (Poisson): A=0 for problematic ones.
            -> TODO: ignore stuff that sEXTRACTOR finds that is not associated to real targets.
            
            
            
            cood that goes through the target and don't pass the selection cut, and enter selection PQR.
            
            c++ code does the full integration that makes the selection cut.
            
            '''
            # generate galaxies at positions ********************************************
      
            # this sets the PSF for the tile
            pp = list(params_image_sims.keys())
            if config['index_P0_PSF']:
                if config['index_P0_PSF']=='turb':
                    psf_fwhm = 1.1+(np.random.random(1)*0.4)[0]
                    #if (np.random.random(1)[0]>0.5):
                    #    psf_fwhm = 1.1
                    #else:
                    #    psf_fwhm = 1.5#+(np.random.random(1)*0.4)[0]
                    Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)
                    psf_pars_ = {'pars':[.0, 0., 0., 0., Tpsf, 1.0],'turb':True}
                    
                    
                else:
                    p0_PSF = pp[config['index_P0_PSF']]
                
            else:
                p0_PSF = pp[np.random.randint(0,len(pp),1)[0]]
                
            for real in range(len(x_a)):
                try:
                    # make WCS for the object ***********************
                    cent=(y_a[real],x_a[real])
                    origin = (0.,0.)
                    duv_dxy = np.array([[0.263, 0.],
                                        [0., 0.263]])
                    wcs_ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)

                    # choose galaxy model *****************
                    redoit = True
                    count_repeat = 0 
                    while redoit:
                        if config['index_P0'] and count_repeat<20:
                            p0 = pp[config['index_P0']]
                        else:
                            p0 = pp[np.random.randint(0,len(pp),1)[0]] #len(pp)
                            if do_templates: # OK I virtually need to allow this to be multiple times of the actual pp len. This can be good for blending effects!!
                                p0 = pp[(config['gal_per_tile']*ii_chunk+real)%len(pp)]
                        redoit = True
                        for b, band in enumerate(config['bands']):
                            gp = params_image_sims[p0][band]['gal_pars']
                            if ((gp[2]**2+gp[3]**2)*config['resize_sn']**2>=0.95) or (gp[4]> config['size_treshold']):
                                
                                skip_this_model+=1. 
                                
                                if b == 0:
                                    redoit= True
                                else:
                                    redoit = redoit | True
                                
                            else:
                                if b == 0:
                                    redoit= False
                                else:
                                    redoit = redoit | False
                  
                    # read the galaxy and PSF and randomly rotate the galaxy

                    twotheta = 2.0 * np.radians(np.random.randint(0,36000)*1./100.)
                    cos2angle = np.cos(twotheta)
                    sin2angle = np.sin(twotheta)


                    if config['resize_sn'] != 1. :
                        gp[3] *= config['resize_sn']
                        gp[2] *= config['resize_sn']


                    galaxy_info = dict()

                    for b in config['bands']:
                        rendering = True
                        count_rendering = 0
                        while rendering:
                            gp = copy.deepcopy(params_image_sims[p0][b]['gal_pars'])
              
                            if config['index_P0_PSF']=='turb':
                                psfp = copy.copy(psf_pars_)
                                #psfp['pars'][4] += np.random.random(1)[0]*0.05* psfp['pars'][4]
                                p0_PSF = np.int(psfp['pars'][4]*100000)
                            else:
                                try:
                                    psfp = params_image_sims[p0_PSF+count_rendering][b]['pfs_params']
                                except:
                                    psfp = params_image_sims[p0_PSF-count_rendering][b]['pfs_params']
                                
                            mm1= gp[2] * cos2angle +  gp[3] * sin2angle
                            mm2= -gp[2] * sin2angle + gp[3] * cos2angle
                            gp[2] = copy.copy(mm1)
                            gp[3] = copy.copy(mm2)

                            galaxy_info[b] = {'gal_p':gp,'psf_p':psfp,'wcs':wcs_,'p0':p0, 'p0_PSF':p0_PSF}

                            if do_templates:
                                mute_p,simulated_psf,jac = render_gal(gp,psfp,wcs_,config['size_tile'], g1 = 0., g2 = 0.,return_PSF=True)
                                mute_m = 1.
                            else:
                                mute_p,simulated_psf,jac = render_gal(gp,psfp,wcs_,config['size_tile'], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)
                                mute_m,jac = render_gal(gp,psfp,wcs_,config['size_tile'],  g1 = config['g1'][1], g2 = config['g2'][1])
                            sv = False
                            if (mute_p is not None) and (mute_m is not None):
                                if do_templates:
                                    tile[b]['image_n'] += mute_p
                                else:
                                    tile[b]['image_p'] += mute_p
                                    tile[b]['image_m'] += mute_m
                                sv = True
                                rendering = False
                                #tile[b]['image_p_noisefree'] += mute_p
                                #tile[b]['image_m_noisefree'] += mute_m
                            else:
                                count_rendering += 1
                                if count_rendering>2:
                                    rendering = False
                                    print ('model rendering failes somehow')
                        if sv:
                            Input_catalog[y_a[real],x_a[real]] = galaxy_info    
                except:
                    if (gp[4]> config['size_treshold']):
                        pass
                    else:
                        print ('problems with model ',p0)
                        
                        
            # render noise ******************************************
            for b in config['bands']:
                if noise_ext:
                        noise = np.random.normal(size = (config['size_tile'],config['size_tile']))*noise_ext*config['noise_factor']
                        tile[b]['noise_level']  = noise_ext*config['noise_factor']

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
                        tile[b]['image_n']+=tile[b]['noise']
                    else:
                        tile[b]['image_p']+=tile[b]['noise']
                        tile[b]['image_m']+=tile[b]['noise']

                            # apply mask
                if not config['maskless']:
                    mask = ( tile[band]['weight']==0. )| (tile[band]['mask']!=0.) 
                    if do_templates:
                        tile[b]['image_n'][mask] = 0. 

                        tile[b]['image_n_noisefree'][mask] = 0.     
                    else:
                        tile[b]['image_p'][mask] = 0. 
                        tile[b]['image_m'][mask] = 0. 

                        tile[b]['image_p_noisefree'][mask] = 0. 
                        tile[b]['image_m_noisefree'][mask] = 0. 

                        
                        
            # detection step ***************************
            if config['mode_detection'] == 'detection':
                detection_cat = dict()

                for b in config['bands']:
                    detection_cat[b] = dict()
                    if do_templates:
                        detection_cat[b]['image_n'], seg_p = sxdes.run_sep(tile[band]['image_n'], config['noise_factor']*noise_ext)

                    else:
                        detection_cat[b]['image_p'], seg_p = sxdes.run_sep(tile[band]['image_p'], config['noise_factor']*noise_ext)
                        detection_cat[b]['image_m'], seg_m = sxdes.run_sep(tile[band]['image_m'], config['noise_factor']*noise_ext)

        
            # measure moments **********************
            ll = np.array(list(Input_catalog.keys()))


            if do_templates:
                loop_keys = ['image_n']
            else:
                loop_keys = ['image_p','image_m']
  
            for im_type in loop_keys:
                if config['mode_detection'] == 'detection':
                    extra_obj_data_fields = [('number', 'i8'),]
                    obj_data = meds.util.get_meds_input_struct(len(detection_cat[config['bands'][0]][im_type]['x']), extra_fields=extra_obj_data_fields)
                    obj_data["id"] = detection_cat[config['bands'][0]][im_type]['number']
                    obj_data["number"] = detection_cat[config['bands'][0]][im_type]['number']
                    obj_data["ra"] = detection_cat[config['bands'][0]][im_type]['x']
                    obj_data["dec"] = detection_cat[config['bands'][0]][im_type]['y']
                    obj_data["box_size"] = get_box_sizes(detection_cat[config['bands'][0]][im_type])

                    len_loop = len(detection_cat[config['bands'][0]][im_type]['y'])

                else:

                    len_loop = len(Input_catalog.keys())


        
                # sorting inputs/detections for templates ******************************************************************************
                # basically we want templates within 2 arcsec to be considered as 1 single obect ***************************************
                if config['mode_detection'] == 'detection':
                    indexes_final = np.array(detection_cat[config['bands'][0]][im_type]['x'] == detection_cat[config['bands'][0]][im_type]['x'])
                else:
                    indexes_final = np.array(list(Input_catalog.keys()))[:,0] == np.array(list(Input_catalog.keys()))[:,0]
               
                if do_templates:
                    if config['mode_detection'] == 'input':
                        ll = np.array(list(Input_catalog.keys()))
                        x = ll[:,0]
                        y = ll[:,1]

                    if config['mode_detection'] == 'detection':
                        x = detection_cat[config['bands'][0]][im_type]['x']
                        y = detection_cat[config['bands'][0]][im_type]['y']

                    
                
                    

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

                
                # check how many inout galaxies have been missed ------
                if (config['mode_detection'] =='detection'):
                    x = detection_cat[config['bands'][0]][im_type]['x']
                    y = detection_cat[config['bands'][0]][im_type]['y']

                    
                    ll = np.array(list(Input_catalog.keys()))
                    x_i = ll[:,0]
                    y_i = ll[:,1]
                    
                    catalog_input = SkyCoord(ra=x_i*uu.arcsec*0.263, dec=y_i*uu.arcsec*0.263)  
                    catalog_1 = SkyCoord(ra=x*uu.arcsec*0.263, dec=y*uu.arcsec*0.263)  
                    idx, d2d, d3d = catalog_input.match_to_catalog_sky(catalog_1, nthneighbor=1) 
                    dist_pix = np.sqrt((x[idx]-x_i)**2+(y[idx]-y_i)**2)*0.263
                    x_nonmatched = x_i[dist_pix>config['radius_blends_templates']]
                    y_nonmatched = y_i[dist_pix>config['radius_blends_templates']]
                    print ()
                    print ('non matched ',len(x_nonmatched))

                        
                        
                # measurement loop ******************************************************************************
                templates_id = []
                print ('')
                ix_ =0
                for ix in np.arange(len_loop)[indexes_final]:
                        
                        # the outer 40 pixels are never used to inject galaxies, so if there's a detection there it shoud be dropped. 
                        if config['mode_detection'] == 'detection':
                            obj_within_good_area = (detection_cat[config['bands'][0]][im_type]['x'][ix]>=40) & (detection_cat[config['bands'][0]][im_type]['x'][ix]<= (config['size_tile']-40)) &  (detection_cat[config['bands'][0]][im_type]['y'][ix]>=40) & (detection_cat[config['bands'][0]][im_type]['y'][ix]<= (config['size_tile']-40))
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


                                    psf_p =  Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['psf_p']
                                    gal_p =  Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['gal_p']
                                    wcs_ = copy.copy(Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['wcs'])
                                    wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0])
                                    p0 = Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['p0']
                                else:
                                    maskx = (np.arange(config['size_tile'])>=ll[ix][0]-20) & (np.arange(config['size_tile'])<ll[ix][0]-20+40)
                                    masky = (np.arange(config['size_tile'])>=ll[ix][1]-20) & (np.arange(config['size_tile'])<ll[ix][1]-20+40) 


                                    psf_p = Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['psf_p']
                                    gal_p = Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['gal_p']
                                    wcs_ = copy.copy(Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['wcs'])
                                    wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky][0])

                                    p0 = Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['p0']


                                image_stamp = tile[band][im_type][masky,:][:,maskx]
                                wt_stamp = tile[band][im_type][masky,:][:,maskx]



                                # render galaxy and psf model *****
                                if do_templates:
                                    _,  psf_image,jac = render_gal(gal_p,psf_p,wcs_,image_stamp.shape[0], g1 = 0., g2 = 0.,return_PSF=True)
                                else:
                                    _,  psf_image,jac = render_gal(gal_p,psf_p,wcs_,image_stamp.shape[0], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)

                                #print (image_stamp.shape,psf_image.shape)
                                images_a.append(image_stamp)
                                wcs_a.append(wcs_)
                                psf_a.append(psf_image)
                                bands_a.append(band)
                                noise_a.append(tile[band]['noise_level'])

                            kds = bfd.multiImage(images_a, (0,0), psf_a, wcs_a, pixel_noiselist = noise_a, bandlist = bands_a ,pad_factor= config['pad_factor'])
                            wt = mc.KSigmaWeight(sigma = config['sigma']) 
                            mul = bfd.MultiMomentCalculator(kds, wt, band_dict = config['band_dict_code'])
                            xyshift, error,msg = mul.recenter()
                            moments = mul
                            
                            
                            mom,meb = moments.get_moment(0,0,returnbands=True)
                            covm_even,covm_odd,covm_even_all,covm_odd_all = moments.get_covariance(returnbands=True)


                            kds_PSF = bfd.multiImage(psf_a, (0,0), [None]*len(psf_a), wcs_a, pixel_noiselist = noise_a, bandlist = bands_a,pad_factor=config['pad_factor'])
                            mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, band_dict = config['band_dict_code'])

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
                                    Wide_g = Image(Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['p0'], meds = [], bands = config['bands'])
                                else:
                                    Wide_g = Image(Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['p0'], meds = [], bands = config['bands'])


                                tab_detections.add_image(Wide_g)
                               
                                tab_detections.images[ix_].moments = mul

                                mom = tab_detections.images[ix_].moments.get_moment(0.,0.)

                                
                                
                                if config['mode_detection'] == 'detection':
                                    tab_detections.images[ix_].p0 = Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['p0']
                                    tab_detections.images[ix_].p0_PSF = Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['p0_PSF']
                                    templates_id.append(Input_catalog[(ll[np.int(idx)][0],ll[np.int(idx)][1])][band]['p0'])
                                if config['mode_detection'] == 'input':
                                    tab_detections.images[ix_].p0 = Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['p0']
                                    tab_detections.images[ix_].p0_PSF = Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['p0_PSF']
                                    templates_id.append(Input_catalog[(ll[np.int(ix)][0],ll[np.int(ix)][1])][band]['p0'])
                                ix_ += 1


                            if not do_templates:
                                if meb[0,0] != meb[0,0]:
                                    print ('NAN ',im_type,dist_pix,newcent,wcs_.xy0 )
                
                                if im_type == 'image_p':
                                    tab_targets.add(mom, xy=newcent,id=ix,number=1,covgal=MomentCovariance(covgal[0],covgal[1]))
                                    tab_targets.p0.append(p0)
                                    tab_targets.p0_PSF.append(p0_PSF)
                                    tab_targets.ra.append(newcent[0])
                                    tab_targets.dec.append(newcent[1])
                                    tab_targets.ra_shift.append(newcent_shift[0])
                                    tab_targets.dec_shift.append(newcent_shift[1])
                                    tab_targets.AREA.append(0.)
                                    
                                    meb_ = np.array([m_.even for m_ in meb])
                                    tab_targets.meb.append(meb_[0,:])
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
                                    tab_targets_m.add(mom, xy=newcent,id=ix,number=1,covgal=MomentCovariance(covgal[0],covgal[1]))
                                    tab_targets_m.p0.append(p0)
                                    tab_targets_m.p0_PSF.append(p0_PSF)
                                    tab_targets_m.ra.append(newcent[0])
                                    tab_targets_m.dec.append(newcent[1])
                                    tab_targets_m.ra_shift.append(newcent_shift[0])
                                    tab_targets_m.dec_shift.append(newcent_shift[1])
                                    
                                    meb_ = np.array([m_.even for m_ in meb])
                                    tab_targets_m.meb.append(meb_[0,:])
                                    tab_targets_m.AREA.append(0.)
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


                # measure the additional stamp for selection purposes [applies only if RANDOM + DETECTION + POISSON] **********************************
                if ((not config['grid_targets']) & (config['poisson']) & (config['mode_detection'] =='detection')):
                    images_a = []
                    wcs_a = []
                    psf_a = []
                    bands_a = []
                    noise_a = []
                    for band in config['bands']:
                        maskx = (np.arange(config['size_tile'])>=x_a_poisson_selection-20) & (np.arange(config['size_tile'])<x_a_poisson_selection-20+40)
                        masky = (np.arange(config['size_tile'])>=y_a_poisson_selection-20) & (np.arange(config['size_tile'])<y_a_poisson_selection-20+40)
                        psf_p = Input_catalog[(ll[0][0],ll[0][1])][band]['psf_p']
                        gal_p = Input_catalog[(ll[0][0],ll[0][1])][band]['gal_p']
                        wcs_ = copy.copy(Input_catalog[(ll[0][0],ll[0][1])][band]['wcs'])
                        wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky])
                        p0 = Input_catalog[(ll[0][0],ll[0][1])][band]['p0']
                        image_stamp = tile[band][im_type][masky,:][:,maskx]
                        wt_stamp = tile[band][im_type][masky,:][:,maskx]                       


                        images_a.append(image_stamp)
                        wcs_a.append(wcs_)
                        psf_a.append(psf_image)
                        bands_a.append(band)
                        noise_a.append(tile[band]['noise_level'])

                    kds = bfd.multiImage(images_a, (0,0), psf_a, wcs_a, pixel_noiselist = noise_a, bandlist = bands_a ,pad_factor= config['pad_factor'])
                    wt = mc.KSigmaWeight(sigma = config['sigma']) 
                    mul = bfd.MultiMomentCalculator(kds, wt, band_dict = config['band_dict_code'])
                    #xyshift, error,msg = mul.recenter()
                    moments = mul

                    mom,meb, mob = moments.get_moment(0,0,return_bands=True)
                    covm_even,covm_odd,covm_even_all,covm_odd_all = moments.get_covariance(return_bands=True)


                    kds_PSF = bfd.multiImage(psf_a, (0,0), [None]*len(psf_a), wcs_a, pixel_noiselist = noise_a, bandlist = bands_a,pad_factor=config['pad_factor'])
                    mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, band_dict = config['band_dict_code'])

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


                    tab_targets.add(mom*0., xy=newcent,id=ix,number=1,covgal=covgal)
                    tab_targets.p0.append(p0)
                    tab_targets.p0_PSF.append(p0_PSF)
                    tab_targets.ra.append(newcent[0])
                    tab_targets.dec.append(newcent[1])
                    tab_targets.ra_shift.append(newcent_shift[0])
                    tab_targets.dec_shift.append(newcent_shift[1])
                    tab_targets.AREA.append(AREA)

                    tab_targets.meb.append(meb[0,:]*0.)
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

                    try:
                        tab_targets.des_id.append(params_image_sims[p0]['des_id'])
                        tab_targets.photoz.append(params_image_sims[p0]['photoz'])
                    except:
                        pass

                    tab_targets_m.add(mom*0., xy=newcent,id=ix,number=1,covgal=covgal)
                    tab_targets_m.p0.append(p0)
                    tab_targets_m.p0_PSF.append(p0_PSF)
                    tab_targets_m.ra.append(newcent[0])
                    tab_targets_m.dec.append(newcent[1])
                    tab_targets_m.ra_shift.append(newcent_shift[0])
                    tab_targets_m.dec_shift.append(newcent_shift[1])
                    tab_targets_m.AREA.append(AREA)

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
                # Add missed stamps for selection purposes [applies if DETECTION and non POISSON] ********************************** 
                if ((not config['poisson']) & (config['mode_detection'] =='detection')):
                    for iii in range(len(x_nonmatched)):
                                               
                        images_a = []
                        wcs_a = []
                        psf_a = []
                        bands_a = []
                        noise_a = []
                        for band in config['bands']:
                            maskx = (np.arange(config['size_tile'])>=x_nonmatched[iii]-20) & (np.arange(config['size_tile'])<x_nonmatched[iii]-20+40)
                            masky = (np.arange(config['size_tile'])>=y_nonmatched[iii]-20) & (np.arange(config['size_tile'])<y_nonmatched[iii]-20+40)
                            psf_p = Input_catalog[(ll[0][0],ll[0][1])][band]['psf_p']
                            gal_p = Input_catalog[(ll[0][0],ll[0][1])][band]['gal_p']
                            wcs_ = copy.copy(Input_catalog[(ll[0][0],ll[0][1])][band]['wcs'])
                            wcs_.xy0 = wcs_.xy0 - (np.arange(config['size_tile'])[maskx][0],np.arange(config['size_tile'])[masky])
                            p0 = Input_catalog[(ll[0][0],ll[0][1])][band]['p0']
                            image_stamp = tile[band][im_type][masky,:][:,maskx]
                            wt_stamp = tile[band][im_type][masky,:][:,maskx]                       


                            images_a.append(image_stamp)
                            wcs_a.append(wcs_)
                            psf_a.append(psf_image)
                            bands_a.append(band)
                            noise_a.append(tile[band]['noise_level'])

                        kds = bfd.multiImage(images_a, (0,0), psf_a, wcs_a, pixel_noiselist = noise_a, bandlist = bands_a ,pad_factor= config['pad_factor'])
                        wt = mc.KSigmaWeight(sigma = config['sigma']) 
                        mul = bfd.MultiMomentCalculator(kds, wt, band_dict = config['band_dict_code'])
                        xyshift, error,msg = mul.recenter()
                        moments = mul

                        mom,meb, mob = moments.get_moment(0,0,return_bands=True)
                        covm_even,covm_odd,covm_even_all,covm_odd_all = moments.get_covariance(return_bands=True)


                        kds_PSF = bfd.multiImage(psf_a, (0,0), [None]*len(psf_a), wcs_a, pixel_noiselist = noise_a, bandlist = bands_a,pad_factor=config['pad_factor'])
                        mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, band_dict = config['band_dict_code'])

                        newcent = [x_nonmatched[iii],y_nonmatched[iii]]
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
                                                   


                        tab_targets.add(mom, xy=newcent,id=ix,number=1,covgal=covgal)
                        tab_targets.p0.append(p0)
                        tab_targets.p0_PSF.append(p0_PSF)
                        tab_targets.ra.append(newcent[0])
                        tab_targets.dec.append(newcent[1])
                        tab_targets.ra_shift.append(newcent_shift[0])
                        tab_targets.dec_shift.append(newcent_shift[1])
                        tab_targets.AREA.append(-1.)

                        tab_targets.meb.append(meb[0,:]*0.)
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

                        try:
                            tab_targets.des_id.append(params_image_sims[p0]['des_id'])
                            tab_targets.photoz.append(params_image_sims[p0]['photoz'])
                        except:
                            pass

                        tab_targets_m.add(mom, xy=newcent,id=ix,number=1,covgal=covgal)
                        tab_targets_m.p0.append(p0)
                        tab_targets_m.p0_PSF.append(p0_PSF)
                        tab_targets_m.ra.append(newcent[0])
                        tab_targets_m.dec.append(newcent[1])
                        tab_targets_m.ra_shift.append(newcent_shift[0])
                        tab_targets_m.dec_shift.append(newcent_shift[1])
                        tab_targets_m.AREA.append(AREA)

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
                                           

        
        if ((not config['grid_targets']) & (config['poisson']) & (config['mode_detection'] =='detection')):
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

        else:

            path = config['output_folder']+'/templates/'+'/IS_templates__chunk_'+str(ii_chunk)
            

            tab_detections.EFFAREA = len(np.arange(len_loop)[indexes_final])
            
            # we need to save the AreA of the tiles. such thtat the final number density will be the sum of all the areas
            tab_detections.AREA_tile = AREA
            save_obj(path,tab_detections)
            path_A = config['output_folder']+'/templates/'+'/AIS_templates__chunk_'+str(ii_chunk)
            #print ('')
            #print (len(np.arange(len_loop)[indexes_final]),STAMP_SIM)
            save_obj(path_A,[len(np.arange(len_loop)[indexes_final]),AREA,STAMP_SIM])



print ('pre-MPI')
from mpi4py import MPI  
print ('after-MPI')
def make_tiles_tt(output_folder,**config):
    print ('doing something')
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
    
  
    mute = dict()
    for i in range(len(config['band_dict'])):
        mute[config['band_dict'][i][0]] = BandInfo(config['band_dict'][i][1],i)
    config['band_dict_code'] = mute
    
    if 'targets' in config['do']:
        run_count = 0
        while run_count<number_of_runs_targets:
            comm = MPI.COMM_WORLD
            if (run_count+comm.rank) < number_of_runs_targets:
                pipeline_targets(config,params_image_sims, run_count+comm.rank)
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
    
    if 'templates' in config['do']:
        run_count = 0
        while run_count<number_of_runs_templates:
            comm = MPI.COMM_WORLD
            if (run_count+comm.rank) < number_of_runs_templates:
                pipeline_targets(config,params_image_sims, run_count+comm.rank,do_templates=True)
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
    #


        