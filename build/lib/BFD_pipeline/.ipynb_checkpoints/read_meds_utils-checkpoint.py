import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmix.gmix as gmix
import gc
import ngmix
import scipy
from scipy import interpolate
import copy
from scipy.interpolate import CloughTocher2DInterpolator
import coord
import galsim

# ###################################################################
# MASKING 
#
# DES Y6 bit mask flags
# "BPM":          1,  #/* set in bpm (hot/dead pixel/column)        */
# "SATURATE":     2,  #/* saturated pixel                           */
# "INTERP":       4,  #/* interpolated pixel                        */
# "BADAMP":       8,  #/* Data from non-functional amplifier        */
# "CRAY":        16,  #/* cosmic ray pixel                          */
# "STAR":        32,  #/* bright star pixel                         */
# "TRAIL":       64,  #/* bleed trail pixel                         */
# "EDGEBLEED":  128,  #/* edge bleed pixel                          */
# "SSXTALK":    256,  #/* pixel potentially effected by xtalk from  */
#                     #/*       a super-saturated source            */
# "EDGE":       512,  #/* pixel flag to exclude CCD glowing edges   */
# "STREAK":    1024,  #/* pixel associated with streak from a       */
#                     #/*       satellite, meteor, ufo...           */
# "SUSPECT":   2048,  #/* nominally useful pixel but not perfect    */
# "TAPEBUMP": 16384,  #/* tape bumps                                */

spline_interp_flags = np.sum([1,2,16,64,1024,2048])
noise_interp_flags  = np.sum([4,8,128,256,512])
bad_flags = spline_interp_flags | noise_interp_flags



def _get_nearby_good_pixels(image, bad_msk, nbad, buff=4):
    """
    get the set of good pixels surrounding bad pixels.
    Parameters
    ----------
    image: array
        The image data
    bad_msk: bool array
        2d array of mask bits.  True means it is a bad
        pixel
    Returns
    -------
    bad_pix:
        bad pix is the set of bad pixels, shape [nbad, 2]
    good_pix:
        good pix is the set of bood pixels around the bad
        pixels, shape [ngood, 2]
    good_im:
        the set of good image values, shape [ngood]
    good_ind:
        the 1d indices of the good pixels row*ncol + col
    """

    nrows, ncols = bad_msk.shape

    ngood = nbad*(2*buff+1)**2
    good_pix = np.zeros((ngood, 2), dtype=np.int64)
    good_ind = np.zeros(ngood, dtype=np.int64)
    bad_pix = np.zeros((ngood, 2), dtype=np.int64)
    good_im = np.zeros(ngood, dtype=image.dtype)

    ibad = 0
    igood = 0
    for row in range(nrows):
        for col in range(ncols):
            val = bad_msk[row, col]
            if val:
                bad_pix[ibad] = (row, col)
                ibad += 1

                row_start = row - buff
                row_end = row + buff
                col_start = col - buff
                col_end = col + buff

                if row_start < 0:
                    row_start = 0
                if row_end > (nrows-1):
                    row_end = nrows-1
                if col_start < 0:
                    col_start = 0
                if col_end > (ncols-1):
                    col_end = ncols-1

                for rc in range(row_start, row_end+1):
                    for cc in range(col_start, col_end+1):
                        tval = bad_msk[rc, cc]
                        if not tval:

                            if igood == ngood:
                                raise RuntimeError('good_pix too small')

                            # got a good one, add it to the list
                            good_pix[igood] = (rc, cc)
                            good_im[igood] = image[rc, cc]

                            # keep track of index
                            ind = rc*ncols + cc
                            good_ind[igood] = ind
                            igood += 1

    bad_pix = bad_pix[:ibad, :]

    good_pix = good_pix[:igood, :]
    good_ind = good_ind[:igood]
    good_im = good_im[:igood]

    return bad_pix, good_pix, good_im, good_ind

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]






'''
This is a class that reads the MEDS files and saves all the relevant information needed to compute BFD moments

'''
class Image:
    

    def __init__(self, index, meds = [], bands = [], verbose = 0):

        self.index_MEDS = index
        self.image_ID = [m['id'][index] for m in meds]
        self.bands = bands
        self.n_bands = len(bands)
            
        self.xyshift = None
        self.moments = None
        self.flags = 0
        self.MOF_model_rendered = None
        self.verbose = verbose
        
    def Load_MEDS(self, index, meds = []):
        '''
        Loads into memory a MEDS detection - 
        it saves the images/segmap/weightmap/bmask/jacobian and psf for different exposures.
        The MEDS files is usually organised in cutouts (i.e., NxN stamps)
        n.b: First cutout for each band is always the COADD
        '''
        if meds != None:

            self.image_ra = [m['ra'][index] for m in meds]
            self.image_dec = [m['dec'][index] for m in meds]
            self.ncutout = [m['ncutout'][index] for m in meds]
            self.imlist = [m.get_cutout_list(index) for m in meds]
            
            # orig_row,col : original row & col in a given tile.
            self.orig_rowcol = [[ (meds[b]['orig_row'][index][i],meds[b]['orig_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
            
            # row,col in the cutouts. Is generally close to center of the tile.
            self.rowcol = [[meds[b].get_cutout_rowcol(index,i) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
            self.seglist = [m.get_cutout_list(index, type='seg') for m in meds]
            self.wtlist = [m.get_cutout_list(index, type='weight') for m in meds]
            self.masklist = [m.get_cutout_list(index, type='bmask') for m in meds]
            self.jaclist = [m.get_jacobian_list(index) for m in meds]
           
            try:
                self.psf = [m.get_cutout_list(index, type='psf') for m in meds]
            except:
                if self.verbose > 0:
                    print ('No PSF extension in this file')


    def add_MOF_models(self, MOF_models):
        self.MOF_models = MOF_models
        
    def add_PSF_model(self, index, psf = [], psf_type = 'PSFex', bands = []):
        '''
        It allows to compute the PSF model for each exposure and each detection given the PSF solution. 
        It currently works with PSFex only.
        
        psf model should be per-band and per-exposure.
        '''
        assert bands == self.bands, 'PSF bands and image bands do not match (PSF: {0}, image: {1})'.format(bands,self.bands) 
        if psf_type == 'PSFex':
            
            try:
                self.psf =  [[psf[b][i].get_rec(self.rowcol[b][i][0], self.rowcol[b][i][1]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
            except:
                self.psf =  [[psf[b].get_rec(self.rowcol[b][i][0], self.rowcol[b][i][1]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
         
    def subtract_background(self):
        '''
        It re-computes the background in the outermost pixels of the image and subtract it.
        '''
        
        for b in range(len(self.bands)):
            for i in range((self.ncutout[b])):
                # excludes masked pixels and pixels where objects are detected
                mask = (self.seglist[b][i] == 0) & (self.masklist[b][i] == 0) &(self.wtlist[b][i]!=0.)
                maskarr=np.ones(np.shape(mask),dtype='int')
                maskarr[~mask] = 0
                maskarr[3:-3,3:-3]=0
                v = self.imlist[b][i][np.where(maskarr==1)]
                if len(v>50): # at least 50 pixels...
                    correction = np.median(v)
                    self.imlist[b][i] -= correction
                
    def deal_with_bmask(self, use_COADD_only = False):
        def check_mask_and_interpolate(image,bmask):
            
            bmask |= np.rot90(bmask)
            bad_mask = (bmask & bad_flags) != 0
           
            if np.any(bad_mask):
                
                nbad = bad_mask.sum()
                #print ('')
                #print (nbad,bad_mask.shape)
                #print ('')
                bad_pix, good_pix, good_im, good_ind = _get_nearby_good_pixels(image, bad_mask, nbad)
        
                gi, ind = np.unique(good_ind, return_index=True)
                good_pix = good_pix[ind, :]
                
              
                good_im = good_im[ind]
                
                try:
                    img_interp = CloughTocher2DInterpolator(
                    good_pix,
                    good_im,
                    fill_value=0.0,
                        )
                except:
                    return image
                    
                    
                interp_image = image.copy()
                image[bad_mask] = img_interp(bad_pix)
                if interp_image is None:
                    return image
                return image
            else:
                return image


         
        for b in range(len(self.bands)):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[b]            
            for i in range(start, end):  
                self.imlist[b][i] = copy.copy(check_mask_and_interpolate(self.imlist[b][i],self.masklist[b][i]))


    def compute_mfrac(self):
        '''
        It computes the fraction of pixels masked in each exposure. 
        '''
        self.size =  [[len(self.masklist[b][i].flatten()) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        self.mfrac = [[1.*len(np.zeros(self.size[b][i])[(self.wtlist[b][i].flatten()==0.) | (self.masklist[b][i].flatten()!=0.)])/self.size[b][i] for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        self.mfrac_flag = [[True for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
    def check_mfrac(self, limit=0.1, use_COADD_only = False):
        '''
        It checks which bands have exposures with a fraction of pixels masked above the limit
        and return False for the bands without exposures with masked pixels. 
        '''
        self.compute_mfrac()
        bands_not_masked = dict()
        for b in range(len(self.bands)):
            bands_not_masked[self.bands[b]] = True
            if use_COADD_only:
                if self.mfrac[b][0] > limit:
                    bands_not_masked[self.bands[b]] = False
                    self.mfrac_flag[b][0] = False
            else:
                bands_not_masked[self.bands[b]] = False
                for i in range(1, self.ncutout[b]):
                    # at least one exposure needs to make it
                    if self.mfrac[b][i] > limit:
                        self.mfrac_flag[b][i] = False
                    if self.mfrac[b][i] < limit:
                        bands_not_masked[self.bands[b]] = True
                  
        tolist = []
        for b in bands_not_masked.keys():
            if bands_not_masked[b]:
                 tolist.append(b)
        self.bands_not_masked = tolist
        return tolist
    
    def plot(self, use_COADD_only = False):
        '''
        Plots the image in different band/exposures.
        '''
        
        if use_COADD_only:
            max_n_exposures = np.max([np.max(self.ncutout[index_band]) for  index_band in range(self.n_bands)])
            fig, ax = plt.subplots(5, self.n_bands, sharey=True,figsize=(13,13))
            for index_band in range(self.n_bands):
                try:
                    ax[0,index_band].imshow(self.imlist[index_band][0])
                    ax[1,index_band].imshow(self.seglist[index_band][0])
                    ax[2,index_band].imshow(self.masklist[index_band][0])
                    ax[3,index_band].imshow(self.wtlist[index_band][0])
                    try:
                        ax[4,index_band].imshow(self.psf[index_band][0])
                    except:
                        pass
                    ax[0,index_band].set_xticks([])
                    ax[0,index_band].set_yticks([])
                    ax[1,index_band].set_xticks([])
                    ax[1,index_band].set_yticks([])
                    ax[2,index_band].set_xticks([])
                    ax[2,index_band].set_yticks([])
                    ax[3,index_band].set_xticks([])
                    ax[3,index_band].set_xticks([])
                    ax[4,index_band].set_xticks([])
                    ax[4,index_band].set_yticks([])
                except:
                    #print (i)
                    fig.delaxes(ax[0,index_band])
                    fig.delaxes(ax[1,index_band])
                    fig.delaxes(ax[2,index_band])
                    fig.delaxes(ax[3,index_band])
                    fig.delaxes(ax[4,index_band])
                ax[0,index_band].set_ylabel(self.bands[index_band]+' band\n ID '+str(self.image_ID[index_band])+'\n size '+str(self.imlist[index_band][0].shape[0])+','+str(self.imlist[index_band][0].shape[1])+'\n {0:2.2f},{1:2.2f}'.format(self.image_ra[index_band],self.image_dec[index_band] )   )
                ax[1,index_band].set_ylabel(self.bands[index_band]+' band\n seg map ')
                ax[2,index_band].set_ylabel(self.bands[index_band]+' band\n badpix map ')
                ax[3,index_band].set_ylabel(self.bands[index_band]+' band\n weightmap ')
                ax[4,index_band].set_ylabel(self.bands[index_band]+' band\n psf ')  
        else:
            max_n_exposures = np.max([np.max(self.ncutout[index_band]) for  index_band in range(self.n_bands)])
            fig, ax = plt.subplots(self.n_bands*5, max_n_exposures, sharey=True,figsize=(13,32))
            for index_band in range(self.n_bands):
                for i in range(max_n_exposures):
                    try:
                        ax[index_band,i].imshow(self.imlist[index_band][i])
                        ax[3+index_band,i].imshow(self.seglist[index_band][i])
                        ax[6+index_band,i].imshow(self.masklist[index_band][i])
                        ax[9+index_band,i].imshow(self.wtlist[index_band][i])
                        try:
                            ax[12+index_band,i].imshow(self.psf[index_band][i])
                        except:
                            pass
                        ax[index_band,i].set_xticks([])
                        ax[index_band,i].set_yticks([])
                        ax[3+index_band,i].set_xticks([])
                        ax[3+index_band,i].set_yticks([])
                        ax[6+index_band,i].set_xticks([])
                        ax[6+index_band,i].set_yticks([])
                        ax[9+index_band,i].set_xticks([])
                        ax[9+index_band,i].set_xticks([])
                        ax[12+index_band,i].set_xticks([])
                        ax[12+index_band,i].set_yticks([])
                    except:
                        #print (i)
                        fig.delaxes(ax[index_band,i])
                        fig.delaxes(ax[index_band+3,i])
                        fig.delaxes(ax[index_band+6,i])
                        fig.delaxes(ax[index_band+9,i])
                        fig.delaxes(ax[index_band+12,i])
                ax[index_band,0].set_ylabel(self.bands[index_band]+' band\n ID '+str(self.image_ID[index_band])+'\n size '+str(self.imlist[index_band][0].shape[0])+','+str(self.imlist[index_band][1].shape[1])+'\n {0:2.2f},{1:2.2f}'.format(self.image_ra[index_band],self.image_dec[index_band] )   )
                ax[index_band+3,0].set_ylabel(self.bands[index_band]+' band\n seg map ')
                ax[index_band+6,0].set_ylabel(self.bands[index_band]+' band\n badpix map ')
                ax[index_band+9,0].set_ylabel(self.bands[index_band]+' band\n weightmap ')
                ax[index_band+12,0].set_ylabel(self.bands[index_band]+' band\n psf ')
            
    def compute_noise(self, del_list = False):
        
        self.noise_rms =  [[1./np.sqrt(np.median(self.wtlist[b][i][(self.wtlist[b][i]!= 0.) & (self.masklist[b][i] == 0)])) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
        if del_list:
            self.wtlist = None
            self.masklist = None
            gc.collect()
                
            
    def zero_padd_psf(self):
        '''
        it zero-padd the psf solutions such that they have the same size as the images
        '''
        
        for index_band in range(self.n_bands):
                size_x = self.imlist[index_band][0].shape[0]
                size_y = self.imlist[index_band][0].shape[1]
                for exp in range(self.ncutout[index_band])[::-1]:            
                    mute = np.zeros((size_x,size_y))
                    size_psf_x = self.psf[index_band][exp].shape[0]
                    size_psf_y = self.psf[index_band][exp].shape[1]
                    # sometimes the PiFF stamp is larger than the image
                    if size_psf_x>size_x:
                        dx = -np.int((size_x-size_psf_x)/2)
                        self.psf[index_band][exp] = self.psf[index_band][exp][dx:dx+size_x,:][:,dx:dx+size_x]
                    # if the PiFF stamp is smaller than the image
                    elif size_psf_x<size_x:
                        dx = np.int((size_x-size_psf_x)/2)
                        mute[dx:dx+size_psf_x,:][:,dx:dx+size_psf_x] = self.psf[index_band][exp]
                        self.psf[index_band][exp] = mute
      

    def compute_nneighbours(self):
        '''
        it computes how many exposures have >0 two entries in the segmap
        '''
        self.exposures_ubserseg=0
        self.nneighbours =  [[len(np.unique(self.seglist[b][i].flatten())) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        for index_band in range(self.n_bands):
            for exp in range(self.ncutout[index_band]):
                if self.nneighbours[index_band][exp]>2:
                    self.exposures_ubserseg+=1

            
    def make_WCS(self):
        '''
        it makes the BFD wcs 
        '''
        
        self.wcslist = []
        for index_band in range(self.n_bands):
            mute_bands =[]
            for exp in range(self.ncutout[index_band]):  
                jac = self.jaclist[index_band][exp]
                cent=(jac['col0'],jac['row0'])
                origin = (0.,0.)
                duv_dxy = np.array( [ [jac['dudcol'], jac['dudrow']],
                                      [jac['dvdcol'], jac['dvdrow']] ])
                mute_bands.append(bfd.WCS(duv_dxy,xyref=cent,uvref=origin))
            self.wcslist.append(mute_bands)
            
            
    def make_WCS_2objects(self, obj2, band, exp):
        '''
        it makes the BFD wcs for a secon obj Image provided by the user.
        It requires to sepcify the band and the exposure.
        '''
        pos_band = np.arange(self.n_bands)[np.in1d(self.bands,band)][0]
        
        '''
        rowcol are ~ the coordinates of the center of the cutout (i.e. where 'self' is located).
        orig_rowcol are the coordinates wrt the tile, such that (self.orig_rowcol - obj.orig_rowcol) corresponds to the
        coordinate difference between the two objects' center.
        '''

        drow = self.rowcol[pos_band][exp][0] - (self.orig_rowcol[pos_band][exp][0] - obj2.orig_rowcol[pos_band][exp][0])
        dcol = self.rowcol[pos_band][exp][1] - (self.orig_rowcol[pos_band][exp][1] - obj2.orig_rowcol[pos_band][exp][1])
        nbrxyref = (dcol,drow)
        nbruvref = (0,0)
        jac = self.jaclist[pos_band][exp]
        nbrduv_dxy = np.array( [ [jac['dudcol'], jac['dudrow']],
                                 [jac['dvdcol'], jac['dvdrow']]])
        nbrwcs = bfd.WCS(nbrduv_dxy, xyref=nbrxyref, uvref=nbruvref)
        return nbrwcs
            


        


    def compute_moments_multiband(self, sigma = 3, use_COADD_only = False, bands = ['r', 'i', 'z'], 
                                  band_dict = {'r':bfd.BandInfo(0.5,0),'i':bfd.BandInfo(0.5,1),'z':bfd.BandInfo(0.5,2)}, MOF_subtraction = False,pad_factor=1.):
        '''
        Compute moments combining exposures and bands
        '''
        
        imgs = []
        wcss = []
        psfs = []
        noise = []
        bandlist = []
        psfs_None = []
        for  band in (bands):
            index_band = np.arange(len(self.bands))[np.array(np.in1d(self.bands,band))]
            if len(index_band) == 0:
                pass
            else:
                index_band = index_band[0]

                if use_COADD_only:
                    start = 0
                    end = 1
                else:
                    start = 1
                    end = self.ncutout[index_band]
                for exp in range(start, end):  
                    if self.mfrac_flag[index_band][exp]:
                        N = self.imlist[index_band][exp].shape[0]
                        if MOF_subtraction:
                            img = self.imlist[index_band][exp] - self.MOF_model_rendered[index_band][exp]
                        else:
                            img = self.imlist[index_band][exp]
                        imgs.append(img)
                        try:
                            wcss.append(self.wcslist[index_band][exp])
                        except:
                            print (self.wcslist)
                            print (index_band,exp)
                        psfs.append(self.psf[index_band][exp])
                        psfs_None.append(None)
                        try:
                            noise_rms = self.noise_rms[index_band][exp]
                        except:
                            noise_rms = (1./np.sqrt(np.median(self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])))
                        noise.append(noise_rms)
                        bandlist.append(self.bands[index_band])
                        shap = np.array(self.imlist[index_band][exp].shape)/2.
                        
        kds = bfd.multiImage(imgs, (0,0), psfs, wcss, pixel_noiselist = noise, bandlist = bandlist,pad_factor=pad_factor)
        kds_PSF = bfd.multiImage(psfs, (0,0), psfs_None, wcss, pixel_noiselist = noise, bandlist = bandlist,pad_factor=pad_factor)
        
        wt = mc.KSigmaWeight(sigma = sigma) 
        mul = bfd.MultiMomentCalculator(kds, wt, band_dict = band_dict)
        mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, band_dict = band_dict)
        
        self.xyshift, error,msg = mul.recenter()
 
        self.moments = mul
        self.moments_PSF = mul_PSF
        self.band_dict = band_dict
        
        
        return mul
    

    def compute_psf_fwhm(self):
        self.psf_fwhm = []
        for index_band in range(self.n_bands):
            mute_bands =[]
            for exp in range(0, self.ncutout[index_band]):   
                conversion = -np.dot(self.wcslist[index_band][exp].jac,np.ones(2))
                N = self.psf[index_band][exp].shape[0]
                hmx = half_max_x(np.arange(N), self.psf[index_band][exp][:,N//2])
                fwhm = hmx[1] - hmx[0]
                mute_bands.append(np.sqrt((fwhm*conversion[0])**2))
            self.psf_fwhm.append((mute_bands))

    def compute_psf_params(self, use_COADD_only = False, band_dict = None):
        '''
        This routine computes T,g1,g2 for the PSF model using ngmix routines.
        '''
        self.psf_params = []
        def make_ngmix_prior(T, pixel_scale):
            from ngmix import priors, joint_prior

            # centroid is 1 pixel gaussian in each direction
            cen_prior=priors.CenPrior(0.0, 0.0, pixel_scale, pixel_scale)

            # g is Bernstein & Armstrong prior with sigma = 0.1
            gprior=priors.GPriorBA(0.1)

            # T is log normal with width 0.2
            Tprior=priors.LogNormal(T, 0.2)

            # flux is the only uninformative prior
            Fprior=priors.FlatPrior(-10.0, 1.e10)

            prior=joint_prior.PriorSimpleSep(cen_prior, gprior, Tprior, Fprior)
            return prior

        for index_band in range(self.n_bands):
            mute_list = []
            
            if use_COADD_only:
                PSF = copy.copy(self.psf[index_band][0])
            else:
                w = 0.
                PSF = np.zeros_like(self.psf[index_band][1])
                for exp in range(1, self.ncutout[index_band]):  
                    w+=1./(self.noise_rms[index_band][exp]**2)
                    PSF += copy.copy(self.psf[index_band][0])/(self.noise_rms[index_band][exp]**2)
                PSF /= w
                
            end = self.ncutout[index_band]
            for exp in range(0, end):    
                try:
                    # it defines the jacobian in the ngmix format.
                    jac = self.jaclist[index_band][exp]
                    jac['row'] = jac['row0']
                    jac['col'] = jac['col0']
    
                    min_linear_scale = np.min([jac['dudcol'],jac['dvdrow']])
                    jacobian_ngmix = ngmix.Jacobian(**jac)
    
                    # it converts the psf image into a ngmix observation
                    obs = ngmix.Observation(image=self.psf[index_band][exp], jacobian=jacobian_ngmix)
    
                    # make prior
                    T_guess = (self.psf_fwhm[index_band][exp]/ 2.35482)**2 * 2.
                    T = T_guess
                    prior = make_ngmix_prior(T_guess, min_linear_scale)
    
                    # computes g1, g2,T
                    dx, dy, g1, g2, flux = 0., 0., 0., 0., 0.
                    lm_pars = {'maxfev':4000}
                    runner=ngmix.bootstrap.PSFRunner(obs, 'gauss', T, lm_pars, prior=prior)
                    runner.go(ntry=3)
    
                    ngmix_flag = runner.fitter.get_result()['flags']
                    gmix = runner.fitter.get_gmix()
    
                    dx, dy = gmix.get_cen()
                    g1, g2, T = gmix.get_g1g2T()
                    mute_list.append({'g1':g1,'g2':g2,'T':T})
                except:
                    mute_list.append({'g1':None,'g2':None,'T':None})
            self.psf_params.append(mute_list)
            

            
        self.psf_params_average = {'g1': 0. ,'g2': 0., 'T': 0.}
        bands_to_use = self.bands_not_masked
        
        # first one averages over expxosures; then over bands;
        wt = 0.
        for bx, band in enumerate(band_dict):
            index_band = np.arange(len(self.bands))[np.array(np.in1d(self.bands,band))][0]
            if band in bands_to_use:
                mute_g1 = 0.
                mute_g2 = 0.
                mute_T = 0.
                if use_COADD_only:
                    start = 0
                    end = 1
                else:
                    start = 1
                    end = self.ncutout[index_band]
                w = 0.
                for exp in range(start, end):  
                    if self.psf_params[index_band][exp]['g1']!= None:
                        w+=1./(self.noise_rms[index_band][exp]**2)
                        mute_g1 += 1./(self.noise_rms[index_band][exp]**2) * self.psf_params[index_band][exp]['g1']
                        mute_g2 += 1./(self.noise_rms[index_band][exp]**2) * self.psf_params[index_band][exp]['g2']
                        mute_T  += 1./(self.noise_rms[index_band][exp]**2) * self.psf_params[index_band][exp]['T'] 
                mute_g1 /= w
                mute_g2 /= w
                mute_T  /= w
                wt += band_dict[band].weight
                self.psf_params_average['g1'] += band_dict[band].weight*mute_g1 
                self.psf_params_average['g2'] += band_dict[band].weight*mute_g2 
                self.psf_params_average['T']  += band_dict[band].weight*mute_T  
                                    
                    
        self.psf_params_average['g1'] /= wt
        self.psf_params_average['g2'] /= wt
        self.psf_params_average['T']  /= wt

import pyfits as pf
class MOF_table:
    def __init__(self, path, shredder = False):
        '''
        Loads into memory a MOF catalog.

        '''
        self.shredder = shredder

        if shredder:
            self.mof_catalog = pf.open(path)
            self.id_array = self.mof_catalog[1].data['id']
            self.params = self.mof_catalog[1].data['band_pars']
            self.numbands = np.shape(self.params)[2]
            
        else:
            
            try:
                self.mof_catalog = pf.open(path)
                self.cols = self.mof_catalog[1].data.columns.names
                self.id_epoch_array = self.mof_catalog[2].data['id']
                self.id_array = self.mof_catalog[1].data['id']
                self.bdf_mag = self.mof_catalog[1].data['bdf_mag']
                self.bdf_flux = self.mof_catalog[1].data['bdf_flux']
                self.bdf_params = self.mof_catalog[1].data['bdf_pars']
                self.pfs_params = self.mof_catalog[2].data['psf_pars']
                self.bdf_ra = self.mof_catalog[1].data['ra']
                self.bdf_dec = self.mof_catalog[1].data['dec']
                self.numbands = np.shape(self.bdf_mag)[1]
            except:
                for i, pm in enumerate(path):
                    self.mof_catalog = pf.open(pm)
                    self.cols = self.mof_catalog[1].data.columns.names
                    if i==0:

                        self.id_epoch_array = self.mof_catalog[2].data['id']
                        self.id_array = self.mof_catalog[1].data['id']
                        self.bdf_mag = self.mof_catalog[1].data['bdf_mag']
                        self.bdf_flux = self.mof_catalog[1].data['bdf_flux']
                        self.bdf_params = self.mof_catalog[1].data['bdf_pars']
                        self.pfs_params = self.mof_catalog[2].data['psf_pars']
                        self.bdf_ra = self.mof_catalog[1].data['ra']
                        self.bdf_dec = self.mof_catalog[1].data['dec']
                    else:
                        self.id_epoch_array = np.hstack([self.id_epoch_array, self.mof_catalog[2].data['id']])
                        self.id_array = np.hstack([self.id_array ,self.mof_catalog[1].data['id']])
                        self.bdf_mag = np.vstack([self.bdf_mag ,self.mof_catalog[1].data['bdf_mag']])
                        self.bdf_flux = np.vstack([self.bdf_flux, self.mof_catalog[1].data['bdf_flux']])
                        self.bdf_params = np.vstack([self.bdf_params, self.mof_catalog[1].data['bdf_pars']])
                        self.pfs_params = np.vstack([self.pfs_params, self.mof_catalog[2].data['psf_pars']])
                        self.bdf_ra =  np.hstack([self.bdf_ra , self.mof_catalog[1].data['ra']])
                        self.bdf_dec = np.hstack([self.bdf_dec, self.mof_catalog[1].data['dec']])
                    self.numbands = np.shape(self.bdf_mag)[1]           
         
    def select_obj_by_ID(self,ID):
        return np.where(self.id_array == ID)

    def match_epochs_by_ID(self,ID):
        return np.where(self.id_epoch_array == ID)

    def return_model(self, band = 'i', pos = None, pos_epoch = None):
        index_band = self.return_band_val(band,indtostr=False)
        gal_pars = self.bdf_params[pos][0:6]
        gal_pars = np.append(gal_pars,self.bdf_flux[pos][index_band])
        pos_epoch = pos_epoch[index_band]
        psf_pars = self.pfs_params[pos_epoch]
        mof_params = {'gal_pars': gal_pars, 'psf_pars': psf_pars}
        return mof_params

    def return_band_val(self,bandind,indtostr=True):
        if indtostr:
            if self.numbands==5:
                if bandind==0: return 'u'
                if bandind==1: return 'g'
                if bandind==2: return 'r'
                if bandind==3: return 'i'
                if bandind==4: return 'z'
            if self.numbands==4:
                if bandind==0: return 'g'
                if bandind==1: return 'r'
                if bandind==2: return 'i'
                if bandind==3: return 'z'
            if self.numbands==3:
                if bandind==0: return 'J'
                if bandind==1: return 'H'
                if bandind==2: return 'Ks'
        else:
            if self.numbands==5:
                if bandind=='u': return 0
                if bandind=='g': return 1
                if bandind=='r': return 2
                if bandind=='i': return 3
                if bandind=='z': return 4
            if self.numbands==4:
                if bandind=='g': return 0
                if bandind=='r': return 1
                if bandind=='i': return 2
                if bandind=='z': return 3
            if self.numbands==3:
                if bandind=='J':  return 0
                if bandind=='H':  return 1
                if bandind=='Ks': return 2



def render_gal(gal_pars,psf_pars,wcs,shape, g1 = None, g2 = None,return_PSF=False):
    
    
    psf_gmix=gmix.GMix(pars=psf_pars)
    det=np.abs(wcs.getdet()) 
    
    jac=gmix.Jacobian(row=wcs.xy0[1],
                      col=wcs.xy0[0],
                      dudrow=wcs.jac[0,1],
                      dudcol=wcs.jac[0,0],
                      dvdrow=wcs.jac[1,1],
                      dvdcol=wcs.jac[1,0])

    gmix_sky = gmix.GMixModel(gp, model='bdf')

    if (g1  != None) and (g2  != None):
        gmix_sky = gmix_sky.get_sheared(g1,g2)
    gmix_image = gmix_sky.convolve(psf_gmix)
    
    try:
        image = gmix_image.make_image((shape,shape), jacobian=jac, fast_exp=True)
        im_psf = psf_gmix.make_image((shape,shape), jacobian=jac,fast_exp=True)    
    except:
        if return_PSF:
            return None,None
        else:
            
            return None

    if return_PSF:
         return image*det,im_psf*det
    else:
            
        return image*det 
    
    
    
    
    
class DetectionsTable:
    def __init__(self, params):
        '''
        Keeps into memory instances of the Image class (a.k.a detections)
        '''
        self.images = []
        self.params = params
        self.ID_array = []
        self.index_MEDS_array = []
        self.flags_array =[] 
        
    def add_image (self, new_image):
        self.images.append(new_image)
        self.ID_array.append(new_image.image_ID[0])
        self.index_MEDS_array.append(new_image.index_MEDS)
        self.flags_array.append(new_image.flags)
        

    def add_MOF_models(self, MOF_table):
        '''
        It reads MOF parameters from the MOF table for each image and store 
        them into the image instances
        '''
        
        # generate columns of matched positions
        index_to_match = np.arange(len(MOF_table.id_array))
        df1 = pd.DataFrame(data = {'pos': index_to_match} , index = MOF_table.id_array)
        df2 = pd.DataFrame(index = self.ID_array)        
        self.pos = np.array(df2.join(df1).loc[self.ID_array,'pos'])

        # generate columns of matched positions for the epochs, which are in another table
        # and have multiple entries
        index_to_match = np.arange(len(MOF_table.id_epoch_array))
        df1 = pd.DataFrame(data = {'pos': index_to_match} , index = MOF_table.id_epoch_array)
        self.pos_epoch = (df2.join(df1))


        for image, pos  in zip(self.images, self.pos):
            ID = image.image_ID[0]
            MOF_models = dict()
            try:
                pos_epoch = (np.array(self.pos_epoch.loc[ID]).astype(np.int))[:,0]
            except:
                pos_epoch = (np.array(self.pos_epoch.loc[ID]).astype(np.int))[0]
            for band in image.bands:
                try:
                    MOF_models[band] = MOF_table.return_model(band = band, pos = pos, pos_epoch = pos_epoch)
                except:
                    pass
            image.add_MOF_models(MOF_models)

    def render_MOF_models(self, index = 0, render_self = False, render_others = True, use_COADD_only = True, g1 = None, g2 = None):
        '''
        It renders the MOF models for image with index = index.
        '''
        
        if use_COADD_only:
            start = 0
            end = 1
        else:
            start = 1
            end = self.ncutout[index_band]
        
        ii = index
        MOF_model_rendered = []
        for b, band in enumerate(self.images[ii].bands):
            mute = []

            for i in range(start, end):  
                rendered_image = np.zeros((self.images[ii].imlist[b][i].shape[0],self.images[ii].imlist[b][i].shape[0]))
                if render_self:
                    wcs = self.images[ii].make_WCS_2objects(self.images[ii], self.images[ii].bands[b], i)
                    try:
                        rendered_image += render_gal(self.images[ii].MOF_models[band]['gal_pars'],self.images[ii].MOF_models[band]['psf_pars'],wcs,self.images[ii].imlist[b][i].shape[0],model='bdf', g1 = g1, g2 = g2)
                    except:
                        pass
                if render_others:
                    # check first list of neighbours in the segmentation map
                    list_MEDS_indexes = np.unique(self.images[ii].seglist[b][i].flatten())
                    list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=0]
                    list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=self.images[ii].index_MEDS+1]

                    try:
                    #print (list_MEDS_indexes)
                        for MEDS_index in list_MEDS_indexes:
                            jj = np.array(self.index_MEDS_array)[np.in1d(np.array(self.index_MEDS_array),MEDS_index)][0]
                            wcs = self.images[ii].make_WCS_2objects(self.images[jj-1], self.images[ii].bands[b], i)
                            try:
                                rendered_image += render_gal(self.images[jj-1].MOF_models[band]['gal_pars'],self.images[jj-1].MOF_models[band]['psf_pars'],wcs,self.images[ii].imlist[b][i].shape[0],model='bdf', g1 = g1, g2 = g2)
                            except:
                                pass


                        #pass
                    except:
                        pass
                mute.append(rendered_image)
            MOF_model_rendered.append(mute) 
        self.images[ii].MOF_model_rendered = MOF_model_rendered
   
    def get_SN(self, moments = 'Mf', flags = 'All'):
        SN = []
        for i, image in enumerate(self.images):
            get_SN = False
            if flags == 'All':
                get_SN = True
            else:
                if self.images[i].flags == flags:
                    get_SN = True    
            if get_SN:      
                try:
                    mom = image.moments.get_moment(0.,0.)
                    if moments == 'Mf':
                        SN_mute = mom.even[mom.M0]/np.sqrt(image.moments.get_covariance()[0][mom.M0,mom.M0])
                        SN.append(SN_mute)
                    if moments == 'MR':
                        SN_mute = mom.even[mom.MR]/np.sqrt(image.moments.get_covariance()[0][mom.MR,mom.MR])
                        SN.append(SN_mute)
                    if moments == 'M1':
                        SN_mute = mom.even[mom.M1]/np.sqrt(image.moments.get_covariance()[0][mom.M1,mom.M1])
                        SN.append(SN_mute)
                    if moments == 'M2':
                        SN_mute = mom.even[mom.M2]/np.sqrt(image.moments.get_covariance()[0][mom.M2,mom.M2])
                        SN.append(SN_mute)
                        
                    if moments == 'MX':
                        SN_mute = mom.odd[mom.MX]/np.sqrt(image.moments.get_covariance()[0][mom.MX,mom.MX])
                        SN.append(SN_mute)
                    if moments == 'MY':
                        SN_mute = mom.odd[mom.MY]/np.sqrt(image.moments.get_covariance()[0][mom.MY,mom.MY])
                        SN.append(SN_mute)
                except:
                    pass
        return SN
    
    def get_moments(self, moments ='Mf', flags = 'All'):
        M = []
        for i, image in enumerate(self.images):
            get_moments = False
            if flags == 'All':
                get_moments = True
            else:
                if self.images[i].flags == flags:
                    get_moments = True    
            if get_moments:         
    
                try:
                    mom = image.moments.get_moment(0.,0.)
                    if moments == 'Mf':
                        M_mute = mom.even[mom.M0]
                        M.append(M_mute)
                    if moments == 'MR':
                        M_mute = mom.even[mom.MR]
                        M.append(M_mute)
                    if moments == 'M1':
                        M_mute = mom.even[mom.M1]
                        M.append(M_mute)
                    if moments == 'M2':
                        M_mute = mom.even[mom.M2]
                        M.append(M_mute)
                    if moments == 'MX':
                        M_mute = mom.odd[mom.MX]
                        M.append(M_mute)
                    if moments == 'MY':
                        M_mute = mom.odd[mom.MY]
                        M.append(M_mute)
                except:
                    pass
        return M      
        
    def compute_moments(self, sigma, bands = 'All', use_COADD_only = True, flags = 'All', MOF_subtraction = True, band_dict = {'r':bfd.BandInfo(0.5,0),'i':bfd.BandInfo(0.5,1),'z':bfd.BandInfo(0.5,2)}, chunk_range = None,pad_factor = 1.):
        self.params['sigma'] = sigma
        
        if chunk_range == None:
            xx = range(len(self.images)) 
        else:
            xx = range(chunk_range[0],chunk_range[1])
    
    
        for i in xx:
            compute_moments = False
            if flags == 'All':
                compute_moments = True
            else:
                if self.images[i].flags == flags:
                    compute_moments = True    
            if compute_moments: 
                if bands == 'All':
                    bands_to_use = self.images[i].bands_not_masked
                else:
                    bands_to_use = np.array(bands)[np.in1d(np.array(bands), self.images[i].bands_not_masked)]
                #try:
                if 1==1:
                    band_dict_to_use = dict()
                    for bx in bands_to_use:
                        band_dict_to_use[bx]  = band_dict[bx]
       
                    self.images[i].compute_moments_multiband(sigma = sigma, bands = bands_to_use, use_COADD_only = use_COADD_only, MOF_subtraction = MOF_subtraction, band_dict = band_dict_to_use,pad_factor = pad_factor)
                    
                #except:
                #    print ('failed moments computation; object ',i, bands_to_use,self.images[i].bands_not_masked,self.images[i].bands)

            else:
                pass
                #erase info image

                
    def compute_average_psf_fwhm(self):
        count = 0
        psf = []
        for images in self.images:
            count +=1
            psf.append(self.images[i].psf_fwhm)
        psf = np.array(psf)
        self.psf_fwhm_average = np.mean(psf,axis=0)
        return self.psf_fwhm_average
    
    
    def make_targets(self, SN_limit = 0, flags = 'All'):
        self.tab_targets = TargetTable(n = self.params['n'],
                                      sigma = self.params['sigma'],
                                      cov=None)
        for i in range(len(self.images)):
            make_target = False
            if flags == 'All':
                make_target = True
            else:
                if self.images[i].flags == flags:
                    make_target = True    
            if make_target: 
                
                try:
                #    print (i)
                    newcent=np.array([self.images[i].image_ra,self.images[i].image_dec])[:,0]+self.images[i].xyshift/3600.0
                    self.tab_targets.add(self.images[i].moments.get_moment(0.,0.), xy=newcent, id=self.images[i].image_ID[0],number=1,covgal=self.images[i].moments.get_covariance())#,num_exp=len(kdata))#,delta_flux=deltamf,cov_delta_flux=covdeltamf)
                #    print ('ok ',i)
                except:
                    print (i)
                
    
    def make_templates(self, SN_limit = 0, flags = 'All'):
        count = 0
        self.tab_templates = TemplateTable(n = self.params['n'],
                        sigma = self.params['sigma'],
                        sn_min = self.params['sn_min'], 
                        sigma_xy = self.params['sigma_xy'], 
                        sigma_flux = self.params['sigma_flux'], 
                        sigma_step = self.params['sigma_step'], 
                        sigma_max = self.params['sigma_max'],
                        xy_max = self.params['xy_max'])

        for i in range(len(self.images)):
            make_templates = False
            if flags == 'All':
                make_templates = True
            else:
                if self.images[i].flags == flags:
                    make_templates = True    
            if make_templates: 
                try:
                    
                    t = self.images[i].moments.make_templates(self.params['sigma_xy'],sigma_flux=self.params['sigma_flux'], sn_min=self.params['sn_min'], sigma_max=self.params['sigma_max'], 
                    sigma_step=self.params['sigma_step'], xy_max=self.params['xy_max'])
                    if t[0] is None:
                        continue
                    else:   
                        for tmpl in t:
                            count +=1
                            self.tab_templates.add(tmpl)
                except:
                    pass
        print ('number of templates: ',count)
        
    def generate_simulated_images(self,index,config,params_image_sims,use_COADD_only=True,noiseless=False,  maskless = False,size_stamp='auto',index_P0= None,index_P0_PSF=None,noise_factor=1.,noshear = False):

        simulated_imlist_p = copy.deepcopy(self.images[index].imlist)
        simulated_imlist_m = copy.deepcopy(self.images[index].imlist)
        simulated_psf = copy.deepcopy(self.images[index].psf)
        # pick one random model from the list of MOF models.
        pp = list(params_image_sims.keys())
        redoit = True
        while redoit:
            if index_P0 is not None:
                p0 = pp[index_P0]#pp[np.random.randint(0,len(pp),1)[0]]
            else:
                
                p0 = pp[np.random.randint(0,len(pp),1)[0]]
            if index_P0_PSF is not None:
                p0_PSF = pp[index_P0_PSF]#pp[np.random.randint(0,len(pp),1)[0]]
            else:
                p0_PSF = copy.copy(p0)
            redoit = True
            for b, band in enumerate(self.images[index].bands):
                gp = params_image_sims[p0][band]['gal_pars']
                if (gp[2]**2+gp[3]**2)>=0.95:
                    if b == 0:
                        redoit= True
                    else:
                        redoit = redoit | True
                else:
                    if b == 0:
                        redoit= False
                    else:
                        redoit = redoit | False

                        
                        
        for b, band in enumerate(self.images[index].bands):
            
            gp = params_image_sims[p0][band]['gal_pars']
        

        # read the galaxy and PSF and randomly rotate the galaxy
        
        twotheta = 2.0 * np.radians(np.random.randint(0,36000)*1./100.)
        cos2angle = np.cos(twotheta)
        sin2angle = np.sin(twotheta)
    


        for b, band in enumerate(self.images[index].bands):
            
            gp = copy.deepcopy(params_image_sims[p0][band]['gal_pars'])
        
            psfp = params_image_sims[p0_PSF][band]['pfs_params']
            
            #mm1= gp[2] * cos2angle +  gp[3] * sin2angle
            #mm2= -gp[2] * sin2angle + gp[3] * cos2angle
            #gp[2] = copy.copy(mm1)
            #gp[3] = copy.copy(mm2)
#
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.images[index].ncutout[b]
            for i in range(start, end):  
            
                size = (self.images[index].wtlist[b][i].shape[0],self.images[index].wtlist[b][i].shape[1])
                wcs = self.images[index].make_WCS_2objects(self.images[index], self.images[index].bands[b], i)

                if size_stamp!='auto':
                    Norig=size[0]
                    size=(size_stamp,size_stamp)
                    Nnew =size[0]

                    wcs=copy.deepcopy(wcs)
                    wcs.xy0=wcs.xy0-(Norig/2.)+(Nnew/2.)
                  
               
                try:
                    noise = np.random.normal(size = size)*np.sqrt(1.0/self.images[index].wtlist[b][i])
                except:
                    noise = np.random.normal(size = size)*np.sqrt(1.0/np.median(self.images[index].wtlist[b][i]))  

                simulated_imlist_p[b][i],simulated_psf[b][i] = render_gal(gp,psfp,wcs,size[0], g1 = config['g1'][0], g2 = config['g2'][0],return_PSF=True)
                simulated_imlist_m[b][i] = render_gal(gp,psfp,wcs,size[0],  g1 = config['g1'][1], g2 = config['g2'][1])
    
                    
                if simulated_imlist_p[b][i] is None:
                    return None,None,None
                if simulated_imlist_m[b][i] is None:
                    return None,None,None
                    
                
                # add noise
                if not noiseless:
                    simulated_imlist_p[b][i]+=noise*noise_factor
                    simulated_imlist_m[b][i]+=noise*noise_factor
                
                # apply mask
                if not maskless:
                    mask = (self.images[index].wtlist[b][i]==0. )| (self.images[index].masklist[b][i]!=0.)
                    simulated_imlist_p[b][i][mask] = 0.
                    simulated_imlist_m[b][i][mask] = 0.
                
                
                #plt.imshow(simulated_imlist_p[b][i])
                #plt.show()
        return simulated_imlist_p, simulated_imlist_m,simulated_psf
              
        
        
        

        
        
        
        
def save_targets(self):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        col.append(fits.Column(name="moments",format="5E",array=self.moment))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))
        col.append(fits.Column(name="number",format="K",array=self.id))
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
        return self.prihdu,tbhdu
    
    
    
def save_template(self, fitsname,EFFAREA):
        savemoments=[]
        savemoments_dg1 = []
        savemoments_dg2 = []
        savemoments_dmu = []
        savemoments_dg1_dg1 = []
        savemoments_dg1_dg2 = []
        savemoments_dg2_dg2 = []
        savemoments_dmu_dg1 = []
        savemoments_dmu_dg2 = []
        savemoments_dmu_dmu = []
        id = []
        nda = []
        jSuppression = []
        
        

        for i, tmpl in enumerate(self.templates):
            # obtain moments and derivs
            m0 = tmpl.get_moment()
            m1_dg1 = tmpl.get_dg1()
            m1_dg2 = tmpl.get_dg2()
            m1_dmu = tmpl.get_dmu()
            m2_dg1_dg1 = tmpl.get_dg1_dg1()
            m2_dg1_dg2 = tmpl.get_dg1_dg2()
            m2_dg2_dg2 = tmpl.get_dg2_dg2()
            m2_dmu_dg1 = tmpl.get_dmu_dg1()
            m2_dmu_dg2 = tmpl.get_dmu_dg2()
            m2_dmu_dmu = tmpl.get_dmu_dmu()
            # append to each list, merging even and odd moments
            savemoments.append(np.append(m0.even,m0.odd))
            savemoments_dg1.append(np.append(m1_dg1.even,m1_dg1.odd))
            savemoments_dg2.append(np.append(m1_dg2.even,m1_dg2.odd))
            savemoments_dmu.append(np.append(m1_dmu.even,m1_dmu.odd))
            savemoments_dg1_dg1.append(np.append(m2_dg1_dg1.even,m2_dg1_dg1.odd))
            savemoments_dg1_dg2.append(np.append(m2_dg1_dg2.even,m2_dg1_dg2.odd))
            savemoments_dg2_dg2.append(np.append(m2_dg2_dg2.even,m2_dg2_dg2.odd))
            savemoments_dmu_dg1.append(np.append(m2_dmu_dg1.even,m2_dmu_dg1.odd)) 
            savemoments_dmu_dg2.append(np.append(m2_dmu_dg2.even,m2_dmu_dg2.odd)) 
            savemoments_dmu_dmu.append(np.append(m2_dmu_dmu.even,m2_dmu_dmu.odd)) 
            nda.append(tmpl.nda)
            id.append(tmpl.id)
            jSuppression.append(tmpl.jSuppression)
            #del tmpl
            #gc.collect()
        # Create the primary and table HDUs
        col1 = fits.Column(name="id",format="K",array=id)
        col2 = fits.Column(name="moments",format="7E",array=savemoments)
        col3 = fits.Column(name="moments_dg1",format="7E",array=savemoments_dg1)
        col4 = fits.Column(name="moments_dg2",format="7E",array=savemoments_dg2)
        col5 = fits.Column(name="moments_dmu",format="7E",array=savemoments_dmu)
        col6 = fits.Column(name="moments_dg1_dg1",format="7E",array=savemoments_dg1_dg1)
        col7 = fits.Column(name="moments_dg1_dg2",format="7E",array=savemoments_dg1_dg2)
        col8 = fits.Column(name="moments_dg2_dg2",format="7E",array=savemoments_dg2_dg2)
        col9 = fits.Column(name="moments_dmu_dg1",format="7E",array=savemoments_dmu_dg1)
        col10 = fits.Column(name="moments_dmu_dg2",format="7E",array=savemoments_dmu_dg2)
        col11= fits.Column(name="moments_dmu_dmu",format="7E",array=savemoments_dmu_dmu)
        col12= fits.Column(name="weight",format="E",array=nda)
        col13= fits.Column(name="jSuppress",format="E",array=jSuppression)
   
        
        cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13])
        tbhdu = fits.BinTableHDU.from_columns(cols, header=self.hdr)
        tbhdu.header['EFFAREA'] =  EFFAREA*1.
        
        prihdu = fits.PrimaryHDU()

     
        
        thdulist = fits.HDUList([prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
        return
