import numpy as np
import ngmix
import galsim
import joblib
import ngmix.gmix as gmix
import metacal_m
from metacal_m import MetacalFitter, CONFIG
from ngmix.jacobian.jacobian_nb import jacobian_get_vu, jacobian_get_area
import bfd
import copy




def render_gal(gal_pars,psf_pars,wcs,shape, g1 = None, g2 = None,return_PSF=False):
    
    try:
        if psf_pars['turb']:
            psf_gmix = ngmix.GMixModel(pars=psf_pars['pars'], model="turb")
    except:
        psf_gmix=gmix.GMix(pars=psf_pars)
        
    det=np.abs(wcs.getdet()) 
    
    jac=ngmix.jacobian.Jacobian(row=wcs.xy0[1],
                      col=wcs.xy0[0],
                      dudrow=wcs.jac[0,1],
                      dudcol=wcs.jac[0,0],
                      dvdrow=wcs.jac[1,1],
                      dvdcol=wcs.jac[1,0])

    gmix_sky = gmix.GMixModel(gal_pars, model='bdf')

    if (g1  != None) and (g2  != None):
        gmix_sky = gmix_sky.get_sheared(g1,g2)
    gmix_image = gmix_sky.convolve(psf_gmix)
    
    try:
        image = gmix_image.make_image((shape,shape), jacobian=jac, fast_exp=True)
        im_psf = psf_gmix.make_image((shape,shape), jacobian=jac,fast_exp=True)    
        #print ('succ')
    except:
        #print ('fail')
        if return_PSF:
            return None,None
        else:
            
            return None,jac

    if return_PSF:
         return image,im_psf
    else:
            
        return image
    
    
    

    



#from metacal import MetacalFitter, CONFIG

SHEARS = ['noshear', '1p', '1m', '2p', '2m']


def make_sim(*, seed, g1, g2, s2n=1e6):
    print (seed)
    rng = np.random.RandomState(seed=seed)

    gal = galsim.Exponential(half_light_radius=0.5).shear(g1=g1, g2=g2)
    psf = galsim.Gaussian(fwhm=0.9)
    obj = galsim.Convolve([gal, psf])
    dim = 53
    cen = (dim-1)/2
    dither = rng.uniform(size=2, low=-0.5, high=0.5)
    scale = 0.263

    im = obj.drawImage(nx=53, ny=53, offset=dither, scale=scale).array
    nse = np.sqrt(np.sum(im**2)) / s2n
    im += rng.normal(size=im.shape, scale=nse)

    psf_im = psf.drawImage(nx=53, ny=53, scale=scale).array

    jac = ngmix.DiagonalJacobian(
        scale=scale, row=cen+dither[0], col=cen+dither[1]
    )
    psf_jac = ngmix.DiagonalJacobian(
        scale=scale, row=cen, col=cen
    )

    obs = ngmix.Observation(
        image=im,
        weight=np.ones_like(im) / nse**2,
        jacobian=jac,
        bmask=np.zeros_like(im, dtype=np.int32),
        psf=ngmix.Observation(
            image=psf_im,
            jacobian=psf_jac,
        ),
        meta={"orig_row": cen, "orig_col": cen},
    )
    return ngmix.observation.get_mb_obs(obs)


def run_single_sim_pair(seed, s2n=1e6):
    mbobs_plus = make_sim(seed=seed, g1=0.02, g2=0.0, s2n=s2n)
    rng = np.random.RandomState(seed=seed)
    ftr = MetacalFitter(CONFIG, 1, rng)
    ftr.go([mbobs_plus])
    res_p = ftr.result
    if res_p is None:
        return None

    mbobs_minus = make_sim(seed=seed, g1=-0.02, g2=0.0, s2n=s2n)
    rng = np.random.RandomState(seed=seed)
    ftr = MetacalFitter(CONFIG, 1, rng)
    ftr.go([mbobs_minus])
    res_m = ftr.result
    if res_m is None:
        return None

    return res_p, res_m


def _msk_it(*, d, s2n_cut, size_cut, shear=''):
    return (
        (d['mcal_flags'] == 0) &
        (d['mcal_s2n' + shear] > s2n_cut) &
        (d['mcal_T_ratio' + shear] > size_cut)
    )


def measure_g1g2R(*, d, s2n_cut, size_cut):
    msks = {}
    for shear in SHEARS:
        msks[shear] = _msk_it(
            d=d, s2n_cut=s2n_cut, size_cut=size_cut, shear='_' + shear)

    #g1_1p = np.mean(d['mcal_g_1p'][msks['1p'], 0])
    #g1_1m = np.mean(d['mcal_g_1m'][msks['1m'], 0])
    #g2_2p = np.mean(d['mcal_g_2p'][msks['2p'], 1])
    #g2_2m = np.mean(d['mcal_g_2m'][msks['2m'], 1])
    ##'''
    g1_1p_s = np.mean(d['mcal_g_noshear'][msks['1p'], 0])
    g1_1m_s = np.mean(d['mcal_g_noshear'][msks['1m'], 0])
    g2_2p_s = np.mean(d['mcal_g_noshear'][msks['2p'], 1])
    g2_2m_s = np.mean(d['mcal_g_noshear'][msks['2m'], 1])
    
    g1_1p = np.mean(d['mcal_g_1p'][msks['noshear'], 0])
    g1_1m = np.mean(d['mcal_g_1m'][msks['noshear'], 0])
    g2_2p = np.mean(d['mcal_g_2p'][msks['noshear'], 1])
    g2_2m = np.mean(d['mcal_g_2m'][msks['noshear'], 1])
    R11 = (g1_1p+g1_1p_s - g1_1m-g1_1m_s) / 2 / 0.01
    R22 = (g2_2p+g2_2p_s - g2_2m-g2_2m_s) / 2 / 0.01
    #'''
    
    #R11 = (g1_1p - g1_1m) / 2 / 0.01
    #R22 = (g2_2p - g2_2m) / 2 / 0.01
#
    g1 = np.mean(d['mcal_g_noshear'][msks['noshear'], 0])
    g2 = np.mean(d['mcal_g_noshear'][msks['noshear'], 1])

    return g1, g2, R11, R22


def measure_m_c(res_p, res_m,s2n_cut,size_cut):
    g1p, g2p, R11p, R22p = measure_g1g2R(d=res_p, s2n_cut=s2n_cut, size_cut=size_cut)
    g1m, g2m, R11m, R22m = measure_g1g2R(d=res_m, s2n_cut=s2n_cut, size_cut=size_cut)

    m = (g1p - g1m)/(R11p + R11m)/0.02 - 1
    c = (g2p + g2m)/(R22p + R22m)
    return m, c, g1p/R11p,g1m/R11m,g2p/R22p,g2m/R22m


def measure_m_c_bootstrap(res_p, res_m, seed, nboot=100,s2n_cut=10,size_cut=0.5):
    rng = np.random.RandomState(seed=seed)
    marr = []
    carr = []
    g1p_e = []
    g1m_e = []
    g2p_e = []
    g2m_e = []
    for _ in range(nboot):
        inds = rng.choice(res_p.shape[0], size=res_p.shape[0], replace=True)
        m, c,g1p,g1m,g2p,g2m = measure_m_c(res_p[inds], res_m[inds],s2n_cut,size_cut)
        marr.append(m)
        carr.append(c)
        g1p_e.append(g1p)
        g1m_e.append(g1m)
        g2p_e.append(g2p)
        g2m_e.append(g2m)
    m, c,g1p,g1m,g2p,g2m = measure_m_c(res_p, res_m,s2n_cut,size_cut)
    
    str1 = "g1p: {0:2.5f} +/- {1:2.5f} ".format(g1p, np.std(g1p_e))
    str2 = "g1m: {0:2.5f} +/- {1:2.5f} ".format(g1m, np.std(g1m_e))
    str3 = "g2p: {0:2.5f} +/- {1:2.5f} ".format(g2p, np.std(g2p_e))
    str4 = "g2m: {0:2.5f} +/- {1:2.5f} ".format(g2m, np.std(g2m_e))

    return m, np.std(marr), c, np.std(carr),str1,str2,str3,str4







def run_single_sim_pair3(seed,parss,parss_PSF,resize_sn = 1.,n=1.,g1=[0.02,0],g2=[-0.02,0],turb=False):
    size_tile = 40
    wcs = {'row0': (size_tile)/2.,
     'col0': (size_tile)/2.,
     'dudrow': 0.,
     'dudcol': 0.263,
     'dvdrow': 0.263,
     'dvdcol': 0.}
    
    if not resize_sn:
        resize_sn = 1
    

    bands = ['r']
    

    y_a = [20]
    x_a = [20]
    real = 0
    

    for b, band in enumerate(bands):
        gp = parss[band]['gal_pars']

    # read the galaxy and PSF and randomly rotate the galaxy

    twotheta = 2.0 * np.radians(np.random.randint(0,36000)*1./100.)
    cos2angle = np.cos(twotheta)
    sin2angle = np.sin(twotheta)

 
    if resize_sn != 1. :
        gp[3] *= resize_sn
        gp[2] *= resize_sn
            
            
    b = bands[0]
    # make WCS for the object ***********************
    cent=(y_a[real],x_a[real])
    origin = (0.,0.)
    duv_dxy = np.array( [ [wcs['dudcol'], wcs['dudrow']],
                          [wcs['dvdcol'], wcs['dvdrow']] ])
    wcs_ = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)
                
    galaxy_info = dict()
    

    gp = copy.deepcopy(parss[b]['gal_pars'])
  
    if turb:
        psfp = {'pars':[.0, 0., 0., 0., parss_PSF/100000, 1.0],'turb':True}
    else:
        psfp = parss_PSF[b]['pfs_params']

    mm1= gp[2] * cos2angle +  gp[3] * sin2angle
    mm2= -gp[2] * sin2angle + gp[3] * cos2angle
    gp[2] = copy.copy(mm1)
    gp[3] = copy.copy(mm2)
  
    galaxy_info[b] = {'gal_p':gp,'psf_p':psfp,'wcs':wcs_}
    
    mute_p,simulated_psf = render_gal(gp,psfp,wcs_,size_tile, g1 = g1[0], g2 = g2[0],return_PSF=True)
    mute_m = render_gal(gp,psfp,wcs_,size_tile, g1 =g1[1], g2 = g2[1])

    noise = np.random.normal(size = (size_tile,size_tile))*n
    wmp = np.ones((size_tile,size_tile))*1./n**2
    mute_p+=noise
    mute_m+=noise
    

    jac_ = ngmix.jacobian.Jacobian(row= (size_tile)/2.,
                      col= (size_tile)/2.,
                      dudrow=wcs_.jac[0,1],
                      dudcol=wcs_.jac[0,0],
                      dvdrow=wcs_.jac[1,1],
                      dvdcol=wcs_.jac[1,0])


    psf_o = ngmix.Observation(simulated_psf,
                jacobian=jac_)
    
    obs_p = ngmix.Observation(mute_p,
        weight=wmp,
        bmask=np.zeros_like(mute_p, dtype=np.int32),
        #noise=noise,
        meta={"orig_row": cent[0], "orig_col": cent[1]},
        jacobian=jac_,
        psf=psf_o)
    
    obs_m= ngmix.Observation(mute_m,
        weight=wmp,
        bmask=np.zeros_like(mute_p, dtype=np.int32),
        #noise=noise,
        meta={"orig_row": cent[1], "orig_col": cent[1]},
        jacobian=jac_,
        psf=psf_o)
    
    obs_p = ngmix.observation.get_mb_obs(obs_p)
    obs_m = ngmix.observation.get_mb_obs(obs_m)
    

    
    rng = np.random.RandomState(seed=seed)
    ftr = MetacalFitter(CONFIG, 1, rng)
    ftr.go([obs_p])
    res_p = ftr.result
    if res_p is None:
        return None,None

    rng = np.random.RandomState(seed=seed)
    ftr = MetacalFitter(CONFIG, 1, rng)
    ftr.go([obs_m])
    res_m = ftr.result
    return res_p, res_m