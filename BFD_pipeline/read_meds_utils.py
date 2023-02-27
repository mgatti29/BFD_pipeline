import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
import glob
import numpy as np
import pandas as pd
import astropy.io.fits as fits
import ngmix.gmix as gmix

import gc
import ngmix
import scipy
from scipy import interpolate
import copy
from scipy.interpolate import CloughTocher2DInterpolator
import coord
import galsim
import astropy.io.fits as fits
import frogress
import shredder
import esutil as eu



spline_interp_flags = np.sum([1,2,16,64,1024,2048])
noise_interp_flags  = np.sum([4,8,128,256,512])
bad_flags = spline_interp_flags | noise_interp_flags


def check_mask_and_interpolate(image,bmask):
    

    
    bmask |= np.rot90(bmask)
    bad_mask = (bmask & bad_flags) != 0

    self.images[ii].MOF_model_all_rendered
    
    '''
    nbad = 0
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
            return image,0


        interp_image = image.copy()
        interp_image = image.copy()
        interp_image[bad_mask] = img_interp(bad_pix)

        if interp_image is None:
            return image,0
        return interp_image,nbad
    else:
        return image,0
    '''
    
    
        
        

   
    
            
class BandInfo:
    def __init__(self, weight=1., index=0):
        self.weight = float(weight)
        self.index = int(index)

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
        try:
            if len(meds)>0:
                self.image_ID = [m['id'][index] for m in meds]
            else:
                 self.image_ID = [index]
        except:
            self.image_ID = [index]
            
        self.bands = bands
        self.n_bands = len(bands)
            
        self.xyshift = None
        self.moments = None
        self.flags = 0
        self.MOF_model_rendered = None
        self.MOF_model_all_rendered = None
        self.MOF_index = None
        self.verbose = verbose
        
    def Load_MEDS_fast(self, index, meds = [],load_seglist = True):
        '''
        Loads into memory a MEDS detection - 
        it saves the images/segmap/weightmap/bmask/jacobian and psf for different exposures.
        The MEDS files is usually organised in cutouts (i.e., NxN stamps)
        n.b: First cutout for each band is always the COADD
        '''
        if meds != None:

            #self.image_ra = [m['ra'][index] for m in meds]
            #self.image_dec = [m['dec'][index] for m in meds]
            self.ncutout = [m['ncutout'][index] for m in meds]
            #self.imlist = [m.get_cutout_list(index) for m in meds]
            #
            ## orig_row,col : original row & col in a given tile.
            self.orig_rowcol = [[ (meds[b]['orig_row'][index][i],meds[b]['orig_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
            self.orig_start_rowcol = [[ (meds[b]['orig_start_row'][index][i],meds[b]['orig_start_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
          
            
    
            # row,col in the cutouts. Is generally close to center of the tile.
            #self.rowcol = [[meds[b].get_cutout_rowcol(index,i) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
            #self.seglist = [m.get_cutout_list(index, type='seg') for m in meds]
            #self.wtlist = [m.get_cutout_list(index, type='weight') for m in meds]
            #self.masklist = [m.get_cutout_list(index, type='bmask') for m in meds]
            self.jaclist = [m.get_jacobian_list(index) for m in meds]
            if load_seglist:
                self.seglist = [m.get_cutout_list(index, type='seg') for m in meds]
            
           #
            #try:
            #    self.psf = [m.get_cutout_list(index, type='psf') for m in meds]
            #except:
            #    if self.verbose > 0:
            #        print ('No PSF extension in this file')
            #        
                    
                    
                    
    def Load_MEDS(self, index, meds = [],exp_list = None):
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
            self.orig_start_rowcol = [[ (meds[b]['orig_start_row'][index][i],meds[b]['orig_start_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
     
            # row,col in the cutouts. Is generally close to center of the tile.
            self.rowcol = [[meds[b].get_cutout_rowcol(index,i) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
            self.seglist = [m.get_cutout_list(index, type='seg') for m in meds]
            self.wtlist = [m.get_cutout_list(index, type='weight') for m in meds]
            self.masklist = [m.get_cutout_list(index, type='bmask') for m in meds]
            self.jaclist = [m.get_jacobian_list(index) for m in meds]
            
            #get ccd numbers ----
            self.ccd_name =  [[ 1000*b+np.int((meds[b]._image_info['image_path'][meds[b]['file_id'][index][i]]).split('_')[2].strip('c')) for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))] 
            
            self.expnum =  [[ np.int((meds[b]._image_info['image_path'][meds[b]['file_id'][index][i]]).split('_')[0].strip('red/D00')) for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))] 
          


            if exp_list is not None:
                self.expnum_ccd =  [[ (self.ccd_name[b][i-1]-1000*b)+100*self.expnum[b][i-1]  for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))] 

                #shrink explist        
                self.explist =  [[ ~((self.expnum_ccd[b][i-1] in exp_list[self.bands[b]]['exp']) & (self.expnum[b][i-1] in exp_list['all']['exp']))  for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))]
            else:
                self.explist = None
                
            

            # save DESDM coordinates ----
            self.DESDM_coadd_x = [m['input_row'][index] for m in meds]
            self.DESDM_coadd_y = [m['input_col'][index] for m in meds]
            
            

           
            try:
                self.psf = [m.get_cutout_list(index, type='psf') for m in meds]
            except:
                if self.verbose > 0:
                    print ('No PSF extension in this file')


                    
                    

    def make_false_stamp(self,use_COADD_only=True):
        for  band in (self.bands):
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
                    self.imlist[index_band][exp]  = np.random.normal(0,1,self.imlist[index_band][exp].shape[0]*self.imlist[index_band][exp].shape[1]).reshape(self.imlist[index_band][exp].shape[0],self.imlist[index_band][exp].shape[1])*self.wtlist[index_band][exp]
                    self.psf[index_band][exp]  = np.zeros_like(self.psf[index_band][exp])
                    
                    psfn_ = np.zeros_like(self.psf[index_band][exp])        
                    psfn_[len(self.psf[index_band][exp])//2,len(self.psf[index_band][exp])//2] = 1 
                    self.psf[index_band][exp] = psfn_


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
        
        bkg_tot = 0
        count = 0
        len_v = 0
        for b in range(len(self.bands)):
            for i in range((self.ncutout[b])):
                
                
                    seg = copy.deepcopy(self.seglist[b][i])
                    segg0 = copy.deepcopy(self.seglist[b][i])
                    for ii in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                        seg += (np.roll(segg0, ii, axis = 0) + np.roll(segg0, ii, axis = 1))

                    mask__ = copy.deepcopy(self.masklist[b][i] )
                    segg0 = copy.deepcopy(self.masklist[b][i] )
                    for ii in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                        mask__ += (np.roll(segg0, ii, axis = 0) + np.roll(segg0, ii, axis = 1))


                    mask = (seg == 0) & (mask__ == 0) 
                    maskarr=np.ones(np.shape(mask),dtype='int')
                    maskarr[~mask] = 0
                    uu = 6
                    maskarr[uu:-uu,uu:-uu]=0
                    v = self.imlist[b][i][np.where(maskarr==1)]

                    if len(v)>50: 
                        correction = np.median(v)
                        self.imlist[b][i] -= correction
                        bkg_tot += correction
                        count += 1
                        len_v = len(v)               
                        
        if count ==0:
            return bkg_tot,100000000000000
        else:
            return bkg_tot/count,len_v/count
                        
                        
                
                
                
    def deal_with_bmask(self, use_COADD_only = False):        
        for b in range(len(self.bands)):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[b]            
            for i in range(start, end):  
                    
                bmask = copy.deepcopy(self.masklist[b][i])
                bmask |= np.rot90(self.masklist[b][i])
    
                try:
                    self.imlist[b][i][bmask==0] += self.images[ii].MOF_model_all_rendered[b][i][bmask==0]
                except:
                    pass
                s0 = self.MOF_model_all_rendered[b][i].shape[0]
                
                self.imlist[b][i][(bmask==0) & (self.wtlist[b][i]!=0.)] += (np.random.normal(size = (s0,s0))[(bmask==0) & (self.wtlist[b][i]!=0.)]/np.sqrt(self.wtlist[b][i])[(bmask==0) & (self.wtlist[b][i]!=0.)])
                                                                
                    
                #self.imlist[b][i],_ = copy.copy(check_mask_and_interpolate(self.imlist[b][i],self.masklist[b][i]))


    def compute_mfrac(self):
        import timeit
        '''
        It computes the fraction of pixels masked in each exposure. 
        '''
        
        
        st = timeit.default_timer()
        self.size =  [[len(self.masklist[b][i].flatten()) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        #self.mfrac = [[1.*len(np.zeros(self.size[b][i])[(self.wtlist[b][i].flatten()==0.) | (self.masklist[b][i].flatten()!=0.)])/self.size[b][i] for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        self.mfrac_flag = [[True for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        #ed = timeit.default_timer()
        #
        
        
        # gaussian aperture ****
        psf_fwhm = 2.
        Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)

        psf_pars = [0.0, 0.0, 0.0, 0.00, Tpsf, 1.0]
        psf_gmix = ngmix.GMixModel(pars=psf_pars, model="turb")

        
        
        self.mfrac = [[np.sum((psf_gmix.make_image((self.imlist[b][i].shape), jacobian=ngmix.jacobian.Jacobian(row=self.wcslist[b][i].xy0[1],
                      col=self.wcslist[b][i].xy0[0],
                      dudrow=self.wcslist[b][i].jac[0,1],
                      dudcol=self.wcslist[b][i].jac[0,0],
                      dvdrow=self.wcslist[b][i].jac[1,1],
                      dvdcol=self.wcslist[b][i].jac[1,0]),fast_exp=True)).flatten()[((self.wtlist[b][i].flatten()==0.) | (self.masklist[b][i].flatten()!=0.))])/np.sum((psf_gmix.make_image((self.imlist[b][i].shape), jacobian=ngmix.jacobian.Jacobian(row=self.wcslist[b][i].xy0[1],
                      col=self.wcslist[b][i].xy0[0],
                      dudrow=self.wcslist[b][i].jac[0,1],
                      dudcol=self.wcslist[b][i].jac[0,0],
                      dvdrow=self.wcslist[b][i].jac[1,1],
                      dvdcol=self.wcslist[b][i].jac[1,0]),fast_exp=True)).flatten()) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
   
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
                for i in range(1, self.ncutout[b]):  #MG *******+
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
            
    def compute_noise(self, noise_factor=1.,noise_ext=False,del_list = False):
        #print (self.bands)
        #print (noise_factor)
        #print (self.wtlist[0][0])
        #print (self.masklist[0][0])
        #print (self.ncutout)
        if noise_ext:
            self.noise_rms =  [[noise_factor*noise_ext for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        else:
            self.noise_rms =  [[noise_factor*np.sqrt(1./np.median(self.wtlist[b][i][(self.wtlist[b][i]!= 0.) & (self.masklist[b][i] == 0)])) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
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
            
            
    def make_WCS_2objects(self, obj2, band, exp,return_shift=False):
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
        if return_shift:
            return nbrwcs,nbrxyref
        else:
            return nbrwcs
            


        


    def compute_moments_multiband(self, sigma = 3, use_COADD_only = False, bands = ['r', 'i', 'z'], 
                                  band_dict = {'r':BandInfo(0.5,0),'i':BandInfo(0.5,1),'z':BandInfo(0.5,2)}, MOF_subtraction = False,pad_factor=1.,filter_='KBlackmanHarris'):
        '''
        Compute moments combining exposures and bands
        '''
        
        imgs = []
        wcss = []
        wcss_psf = []
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
                    compute = False
                    if self.mfrac_flag[index_band][exp]:
                        
                        if self.explist is not None:
                            if self.explist[index_band][exp-1]:
                                compute = True
                        else:
                            compute = True
                    if compute:
                                
                        
                        N = self.imlist[index_band][exp].shape[0]
                        if MOF_subtraction:
                            img = self.imlist[index_band][exp] - self.MOF_model_rendered[index_band][exp]
                        else:
                            img = self.imlist[index_band][exp]
                        imgs.append(img)
                 
                        
           
                        wcss.append(self.wcslist[index_band][exp])
                            

                        

                        psfs.append(self.psf[index_band][exp])
                        
                        psfn_ = np.zeros_like(self.psf[index_band][exp])
                        # make a wcs for the PSF
                        origin = (0.,0.)
                        cent = (len(psfn_)//2,len(psfn_)//2)
                        duv_dxy = np.array([[self.wcslist[index_band][exp].jac[0,0], self.wcslist[index_band][exp].jac[0,1]],
                                            [self.wcslist[index_band][exp].jac[1,0], self.wcslist[index_band][exp].jac[1,1]]])
                        wcs_psf = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)
                        wcss_psf.append(wcs_psf)
                       
                        
                        psfn_[len(psfn_)//2,len(psfn_)//2] = 1.
                        psfs_None.append(psfn_)
                        try:
                       
                            noise_rms = self.noise_rms[index_band][exp]
                        except:
                            noise_rms = (1./np.sqrt(np.median(self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])))
                        #print ('pixel_noise ',noise_rms)
                        #print ('im_noise ',np.std(self.imlist[index_band][exp]))
                        noise.append(noise_rms)
                        bandlist.append(self.bands[index_band])
                        shap = np.array(self.imlist[index_band][exp].shape)/2.
                        
        kds = bfd.multiImage(imgs, (0,0), psfs, wcss, pixel_noiselist = noise, bandlist = bandlist,pad_factor=pad_factor)
        
        #
        kds_PSF = bfd.multiImage(psfs, (0,0), psfs_None, wcss_psf, pixel_noiselist = noise, bandlist = bandlist,pad_factor=pad_factor)
        
        #wt = mc.KSigmaWeight(sigma = sigma) 
        if filter_ == 'KBlackmanHarris':
            wt = mc.KBlackmanHarris(sigma = sigma) 
        else:
            wt = mc.KSigmaWeight(sigma = sigma) 
            
        mul = bfd.MultiMomentCalculator(kds, wt, bandinfo = band_dict)
        mul_PSF = bfd.MultiMomentCalculator(kds_PSF, wt, bandinfo = band_dict)
        
        xyshift_PSF, error_PSF,msg  = mul_PSF.recenter()
        
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
                    jacobian_ngmix = ngmix.jacobian.Jacobian(**jac)
    
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
            print (path)
            self.mof_catalog = fits.open(path)
            self.id_array = self.mof_catalog[1].data['id']
            self.params = self.mof_catalog[1].data['band_pars']
            self.PSF_params = self.mof_catalog[1].data['band_psf_pars'] #X , 60, 5
            self.PSF_size = self.mof_catalog[1].data['band_psf_T']
            self.flux = self.mof_catalog[1].data['band_flux']
            self.numbands = self.mof_catalog[1].data['band_pars'].shape[2]
            self.shredder = True
            self.tilename = self.mof_catalog[1].data['tilename']
            self.id_epoch_array = self.mof_catalog[1].data['id']
  
            
        else:
            self.shredder = False
            
            
            try:
                self.mof_catalog = fits.open(path)
                
                try:
                    self.des_id = self.mof_catalog[1].data['des_id']
                    self.mag_i =  self.mof_catalog[1].data['mag_i']
                    #self.mag_i =  self.mof_catalog[1].data['mag_r']
                    #self.mag_i =  self.mof_catalog[1].data['mag_z']
                    self.photoz =  self.mof_catalog[1].data['photoz']
                    
                except:
                    pass
            
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
                    self.mof_catalog = fits.open(pm)
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
                    
                    
        self.tilename = self.mof_catalog[1].data['tilename']
        try:
        #if 1==1:
           # print ('mag i')
     
                self.MAG_I = self.mof_catalog[1].data['bdf_mag'][:,-2]

        
                self.MAG_R = self.mof_catalog[1].data['bdf_mag'][:,-3]
      
                self.MAG_Z = self.mof_catalog[1].data['bdf_mag'][:,-1]
      
        except:
            pass
        self.ra = self.mof_catalog[1].data['ra']
        self.dec = self.mof_catalog[1].data['dec']
         
    def select_obj_by_ID(self,ID):
        return np.where(self.id_array == ID)

    def match_epochs_by_ID(self,ID):
        return np.where(self.id_epoch_array == ID)



    def return_model(self, band = 'i', pos = None, pos_epoch = None):
        try:
            index_band = self.return_band_val(band,indtostr=False)
            gal_pars = self.bdf_params[pos][0:6]
            gal_pars = np.append(gal_pars,self.bdf_flux[pos][index_band])
            pos_epoch = pos_epoch[index_band]
            psf_pars = self.pfs_params[pos_epoch]
            mof_params = {'gal_pars': gal_pars, 'psf_pars': psf_pars}
        except:
            index_band = self.return_band_val_s(band,indtostr=False)
            gal_pars = self.params[pos_epoch,:,index_band]
            psf_pars = self.PSF_params[pos_epoch,:,index_band]
            mof_params = {'gal_pars': gal_pars, 'psf_pars': psf_pars}
        return mof_params

    def return_band_val_s(self,bandind,indtostr=True):
        if indtostr:
            if self.numbands==5:
                if bandind==0: return 'g'
                if bandind==1: return 'r'
                if bandind==2: return 'i'
                if bandind==3: return 'z'
                if bandind==4: return 'Y'
        else:
            if self.numbands==5:

                if bandind=='g': return 0
                if bandind=='r': return 1
                if bandind=='i': return 2
                if bandind=='z': return 3
                if bandind=='Y': return 4
            
            
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


            

    
    
    
def render_gal(gal_pars,psf_pars,wcs,shape, g1 = None, g2 = None,return_PSF=False,nbrxyref=None):
    
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

    if len(gal_pars) == 60:
        #print (gal_pars)
        g_ =  copy.deepcopy(gal_pars)
        #print (g_)
        #for i in range(10):
        #    #pass
        #    #print (g_[1+i*6])
        #    g_[1+i*6] = 0.
        #    g_[2+i*6] = 0.
        gmix_sky = ngmix.GMix(pars=g_.reshape(60))
 

    else:
        gmix_sky = gmix.GMixModel(gal_pars, model='bdf')

    if (g1  != None) and (g2  != None):
        gmix_sky = gmix_sky.get_sheared(g1,g2)
    gmix_image = gmix_sky.convolve(psf_gmix)
    
    try:
        im_psf = psf_gmix.make_image((shape,shape), jacobian=jac)#,fast_exp=True)    
        if nbrxyref!= None:
            v, u = jac(nbrxyref[1],nbrxyref[0])
       
            gmix_image.set_cen(v, u)
        
        image = gmix_image.make_image((shape,shape), jacobian=jac)#, fast_exp=True)
        
        
        #print ('succ')
    except:
        #print ('fail')
        if return_PSF:
            return None,None,jac
        else:
            
            return None,jac

    if return_PSF:
         return image,im_psf,jac
    else:
            
        return image,jac
    
    
    
    
    
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

    def insert_image(self,new_image,index):
        self.images.insert(index,new_image)
        self.ID_array.insert(index,new_image.image_ID[0])
        self.index_MEDS_array.insert(index,new_image.index_MEDS)
        self.flags_array.insert(index,new_image.flags)

    def add_MOF_models(self, MOF_table):
        '''
        It reads MOF parameters from the MOF table for each image and store 
        them into the image instances
        '''
        
        
        
        # generate columns of matched positions
        index_to_match = np.arange(len(MOF_table.id_array))
        try:
            data_ = {'pos': index_to_match,
                'TILENAME': (MOF_table.tilename).byteswap().newbyteorder(),
                'MAG_I': (MOF_table.MAG_I).byteswap().newbyteorder(),
                'MAG_R': (MOF_table.MAG_R).byteswap().newbyteorder(),
                'MAG_Z': (MOF_table.MAG_Z).byteswap().newbyteorder(),
                'RA': (MOF_table.ra).byteswap().newbyteorder(),
                'DEC': (MOF_table.dec).byteswap().newbyteorder(),
                }
        except:
            data_ = {'pos': index_to_match,
                'TILENAME': (MOF_table.tilename).byteswap().newbyteorder(),
                'MAG_I': (np.zeros(len(MOF_table.tilename))),#.byteswap().newbyteorder(),
                'MAG_R': (np.zeros(len(MOF_table.tilename))),#.byteswap().newbyteorder(),
                'MAG_Z': (np.zeros(len(MOF_table.tilename))),#.byteswap().newbyteorder(),
                'RA': (MOF_table.ra).byteswap().newbyteorder(),
                'DEC': (MOF_table.dec).byteswap().newbyteorder(),
                }
            
        '''
        try:
            try:
                try:
                    data_ = {'pos': index_to_match,
                        'TILENAME': (MOF_table.tilename).byteswap().newbyteorder(),
                        'MAG_I': (MOF_table.MAG_I).byteswap().newbyteorder(),
                        'MAG_R': (MOF_table.MAG_R).byteswap().newbyteorder(),
                        'MAG_Z': (MOF_table.MAG_Z).byteswap().newbyteorder(),
                        'RA': (MOF_table.ra).byteswap().newbyteorder(),
                        'DEC': (MOF_table.dec).byteswap().newbyteorder(),
                        }
                except:
                    data_ = {'pos': index_to_match,
                        'TILENAME': (MOF_table.tilename).byteswap().newbyteorder(),
                        'MAG_I': (MOF_table.MAG_I),
                        'MAG_R': (MOF_table.MAG_R),
                        'MAG_Z': (MOF_table.MAG_Z),
                        'RA': (MOF_table.ra).byteswap().newbyteorder(),
                        'DEC': (MOF_table.dec).byteswap().newbyteorder(),
                        }
            except:
                    data_ = {'pos': index_to_match,
                    'TILENAME': (MOF_table.tilename).byteswap().newbyteorder(),
                    'MAG_I': (MOF_table.MAG_I).byteswap().newbyteorder(),
                    'RA': (MOF_table.ra).byteswap().newbyteorder(),
                    'DEC': (MOF_table.dec).byteswap().newbyteorder(),
                    }
        except:
            data_ = {'pos': index_to_match,
                'TILENAME': (MOF_table.tilename).byteswap().newbyteorder(),
                'MAG_I': (np.zeros(len(MOF_table.tilename))),#.byteswap().newbyteorder(),
                'MAG_R': (np.zeros(len(MOF_table.tilename))),#.byteswap().newbyteorder(),
                'MAG_Z': (np.zeros(len(MOF_table.tilename))),#.byteswap().newbyteorder(),
                'RA': (MOF_table.ra).byteswap().newbyteorder(),
                'DEC': (MOF_table.dec).byteswap().newbyteorder(),
                }
        
        '''
        df1 = pd.DataFrame(data = data_, index = MOF_table.id_array)
        df2 = pd.DataFrame(index = self.ID_array)        
        self.pos = np.array(df2.join(df1).loc[self.ID_array,'pos'])
        self.MOF_index = np.array(df2.join(df1).loc[self.ID_array].index)
        # here I can add other colums if I think they're interesting. **
        self.TILENAME = np.array(df2.join(df1).loc[self.ID_array,'TILENAME'])
        self.MAG_I = np.array(df2.join(df1).loc[self.ID_array,'MAG_I'])
        self.MAG_R = np.array(df2.join(df1).loc[self.ID_array,'MAG_R'])
        self.MAG_Z = np.array(df2.join(df1).loc[self.ID_array,'MAG_Z'])
        self.RA_DF = np.array(df2.join(df1).loc[self.ID_array,'RA'])
        self.DEC_DF = np.array(df2.join(df1).loc[self.ID_array,'DEC'])
   
        # generate columns of matched positions for the epochs, which are in another table
        # and have multiple entries
        index_to_match = np.arange(len(MOF_table.id_epoch_array))
        df1 = pd.DataFrame(data = {'pos': index_to_match} , index = MOF_table.id_epoch_array)
        self.pos_epoch = (df2.join(df1))


                                  
                                  
        for image, pos,MOF_index,TILENAME,MAG_I,MAG_R,MAG_Z ,ra,dec in zip(self.images, self.pos, self.MOF_index,self.TILENAME,self.MAG_I,self.MAG_R,self.MAG_Z,self.RA_DF,self.DEC_DF):
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
            if MOF_table.shredder:
                image.MOF_model_shredder = True
            else:
                image.MOF_model_shredder = False
                
                
            image.add_MOF_models(MOF_models)
            image.MOF_index = MOF_index
            image.TILENAME = TILENAME
            image.MAG_I = MAG_I
            image.MAG_R = MAG_R
            image.MAG_Z = MAG_Z
            image.RA_DF = ra
            image.DEC_DF = dec

            
            
    def render_MOF_models(self, index = 0, render_self = False, render_others = True, use_COADD_only = True, g1 = None, g2 = None):
        '''
        It renders the MOF models for image with index = index.
        '''
        

        
        ii = index
        MOF_model_rendered = []
        MOF_model_all_rendered = []
        
        
        for b, band in enumerate(self.images[ii].bands):
            
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 0
                end = self.images[ii].ncutout[b]
            
            
            mute = []
            mute_all = []

            for i in range(start, end):  
                rendered_image = np.zeros((self.images[ii].imlist[b][i].shape[0],self.images[ii].imlist[b][i].shape[0]))
                #if render_self:
                if 1 ==1:
                    try:
                        if self.images[ii].MOF_model_shredder :
                            wcs,nbrxyref = self.images[ii].make_WCS_2objects(self.images[ii], self.images[ii].bands[b], i,return_shift=True)
                        else:
                            wcs = self.images[ii].make_WCS_2objects(self.images[ii], self.images[ii].bands[b], i)

                        if self.images[ii].MOF_model_shredder :
                            m__,jac = render_gal(self.images[ii].MOF_models[band]['gal_pars'],self.images[ii].MOF_models[band]['psf_pars'],wcs,self.images[ii].imlist[b][i].shape[0],  g1 = 0., g2 = 0.,nbrxyref=nbrxyref)
                        else:
                            m__,jac = render_gal(self.images[ii].MOF_models[band]['gal_pars'],self.images[ii].MOF_models[band]['psf_pars'],wcs,self.images[ii].imlist[b][i].shape[0],  g1 = 0., g2 = 0.)


                        self_model = m__
                    except:
                        self_model = np.zeros((self.images[ii].imlist[b][i].shape[0],self.images[ii].imlist[b][i].shape[0]))
                if render_others:
                    # check first list of neighbours in the segmentation map. always use segmap coadd
                
                    list_MEDS_indexes = np.unique(self.images[ii].seglist[b][0].flatten())
                    list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=0]
                    list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=self.images[ii].index_MEDS+1]

           
                        
                    for MEDS_index in list_MEDS_indexes:
                        jj = np.array(self.index_MEDS_array)[np.in1d(np.array(self.index_MEDS_array),MEDS_index-1)][0]
                    
                        try:
                        #if 1==1:
                            if self.images[ii].MOF_model_shredder :
                                wcs,nbrxyref = self.images[ii].make_WCS_2objects(self.images[jj], self.images[ii].bands[b], i,return_shift=True)
                            else:
                                wcs = self.images[ii].make_WCS_2objects(self.images[jj], self.images[ii].bands[b], i)
                        except:
                            # sometimes it happens we don't have the exposure for the second object - in that case we don't generate a model.
                            pass
                
                        try:
                            if self.images[ii].MOF_model_shredder :
                                m__,jac = render_gal(self.images[jj].MOF_models[band]['gal_pars'],self.images[ii].MOF_models[band]['psf_pars'],wcs,self.images[ii].imlist[b][i].shape[0],  g1 = 0., g2 = 0.,nbrxyref=nbrxyref)
                            else:
                                m__,jac = render_gal(self.images[jj].MOF_models[band]['gal_pars'],self.images[ii].MOF_models[band]['psf_pars'],wcs,self.images[ii].imlist[b][i].shape[0],  g1 = 0., g2 = 0.)
                            rendered_image += m__
                            #print (m__)
                            #print (rendered_image)
                        except:
                            pass


                        #pass
                    #except:
                    #    pass
                mute.append(rendered_image)
                mute_all.append(rendered_image+self_model)
            MOF_model_rendered.append(mute) 
            MOF_model_all_rendered.append(mute_all) 
        self.images[ii].MOF_model_rendered = MOF_model_rendered
        self.images[ii].MOF_model_all_rendered = MOF_model_all_rendered
      
            
            
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
        
    def compute_moments(self, sigma, bands = 'All', use_COADD_only = True, flags = 'All', MOF_subtraction = True, band_dict = {'r':BandInfo(0.5,0),'i':BandInfo(0.5,1),'z':BandInfo(0.5,2)}, chunk_range = None,pad_factor = 1., filter_ = 'KBlackmanHarris'):
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
                    
                    mask_ = np.in1d(np.array(band_dict['bands']),bands_to_use)
                    
                    band_dict_to_use['bands'] = list(np.array(band_dict['bands'])[mask_])
                    band_dict_to_use['weights'] = list(np.array(band_dict['weights'])[mask_])
                    band_dict_to_use['index'] = list(np.arange(len(band_dict_to_use['weights'] )))
  
       
                    self.images[i].compute_moments_multiband(sigma = sigma, bands = bands_to_use, use_COADD_only = use_COADD_only, MOF_subtraction = MOF_subtraction, band_dict = band_dict_to_use,pad_factor = pad_factor, filter_ = filter_)
                
                
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
                                      sigma = self.params['sigma'])
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
        
    def generate_simulated_images(self,index,config,params_image_sims,use_COADD_only=True,noiseless=False,  maskless = False,size_stamp='auto',index_P0= None,index_P0_PSF=None,noise_factor=1.,noshear = False,count_replica=0,resize_sn=1.,noise_ext = False,g1=[0.,0.],g2=[0.,0.]):
        count_replica_int = copy.copy(count_replica)
        simulated_imlist_p = copy.deepcopy(self.images[index].imlist)
        simulated_imlist_m = copy.deepcopy(self.images[index].imlist)
        simulated_psf = copy.deepcopy(self.images[index].psf)
        # pick one random model from the list of MOF models.
        pp = list(params_image_sims.keys())
        redoit = True
        count_repeat = 0 
        if count_repeat > 20:
            print ('out of 20 reps.')

                    
        while redoit:
            if index_P0 and count_repeat<200:
                p0 = pp[index_P0]#pp[np.random.randint(0,len(pp),1)[0]]
            else:
                p0 = pp[(index+count_replica_int)%len(pp)]
                p0 = pp[np.random.randint(0,len(pp),1)[0]]
            if config['index_P0_PSF']:
                if config['index_P0_PSF']=='turb':
                    
                    psf_fwhm = config['turb'][0]+(np.random.random(1)*config['turb'][1])[0]
                    #if (np.random.random(1)[0]>0.5):
                    #    psf_fwhm = 1.1
                    #else:
                    #    psf_fwhm = 1.5#+(np.random.random(1)*0.4)[0]
                    Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)
                    psf_pars_ = {'pars':[.0, 0., 0., 0., Tpsf, 1.0],'turb':True}
                else:
                    p0_PSF = pp[index_P0_PSF]
            else:
                p0_PSF = copy.copy(p0)
                p0_PSF = pp[np.random.randint(0,len(pp),1)[0]]
                
                
            redoit = True
            for b, band in enumerate(self.images[index].bands):
                gp = params_image_sims[p0][band]['gal_pars']
                
                if (gp[2]**2+gp[3]**2)*resize_sn**2>=0.95:
                    if b == 0:
                        redoit= True
                    else:
                        redoit = redoit | True
                else:
                    if b == 0:
                        redoit= False
                    else:
                        redoit = redoit | False
            count_replica_int += 1
            count_repeat += 1
                        

        # read the galaxy and PSF and randomly rotate the galaxy
        
        twotheta = 2.0 * np.radians(np.random.randint(0,36000)*1./100.)
        cos2angle = np.cos(twotheta)
        sin2angle = np.sin(twotheta)
    

        if resize_sn != 1. :
            gp[3] *= resize_sn
            gp[2] *= resize_sn
        fluxes = []
        for b, band in enumerate(self.images[index].bands):
            
            gp = copy.deepcopy(params_image_sims[p0][band]['gal_pars'])
            fluxes.append(gp[-1])

            if config['index_P0_PSF']=='turb':
                psfp = copy.copy(psf_pars_)
                #psfp['pars'][4] += np.random.random(1)[0]*0.05* psfp['pars'][4]
                p0_PSF = np.int(psfp['pars'][4]*100000)
            else:
                try:
                    psfp = params_image_sims[p0_PSF][band]['pfs_params']
                except:
                    psfp = params_image_sims[p0_PSF][band]['pfs_params']

            mm1= gp[2] * cos2angle +  gp[3] * sin2angle
            mm2= -gp[2] * sin2angle + gp[3] * cos2angle
            gp[2] = copy.copy(mm1)
            gp[3] = copy.copy(mm2)
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
                  
                if noise_ext:
                    noise = np.random.normal(size = size)*noise_ext
                else:
                    try:
                        noise = np.random.normal(size = size)*np.sqrt(1.0/self.images[index].wtlist[b][i])
                    except:
                        noise = np.random.normal(size = size)*np.sqrt(1.0/np.median(self.images[index].wtlist[b][i]))  


                simulated_imlist_p[b][i],simulated_psf[b][i],jac = render_gal(gp,psfp,wcs,size[0], g1 = g1[0], g2 = g2[0],return_PSF=True)

                simulated_imlist_m[b][i],jac  =                    render_gal(gp,psfp,wcs,size[0],  g1 = g1[1], g2 = g2[1])
    
                    
                if simulated_imlist_p[b][i] is None:
                    return None,None,None,None,None,None,None
                if simulated_imlist_m[b][i] is None:
                    return None,None,None,None,None,None,None
                    
                
                # add noise
                if not noiseless:
                    #print (np.std(noise*noise_factor))
                    simulated_imlist_p[b][i]+=noise*noise_factor
                    simulated_imlist_m[b][i]+=noise*noise_factor
                
                # apply mask
                if not maskless:
                    mask = (self.images[index].wtlist[b][i]==0. )| (self.images[index].masklist[b][i]!=0.)
                    simulated_imlist_p[b][i][mask] = 0.
                    simulated_imlist_m[b][i][mask] = 0.
                
                
                #plt.imshow(simulated_imlist_p[b][i])
                #plt.show()
        return simulated_imlist_p, simulated_imlist_m,simulated_psf,p0,p0_PSF,fluxes,jac 
              
        
        
        

        
        
        
        
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
    
    
    
def save_template(self, fitsname,EFFAREA,TIER_NUM):
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
        p0 = []
        p0_PSF = []
        id = []
        class_ = []
        id_gal = []
        nda = []
        jSuppression = []
        
        
        # crazy enough this has to
        k = 0
        for i in frogress.bar(range(len(self.templates))):
            tmpl =self.templates[k]
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
            
            #insert
            
            #self.templates.insert(i,None)
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
            try:
                id_gal.append(tmpl.id_gal)
                if type(tmpl.class_) == np.chararray:
                    class_.append(tmpl.class_[0])
                else:
                    class_.append(tmpl.class_)
            except:
                pass
            p0_PSF.append(tmpl.p0_PSF)
            p0.append(tmpl.p0)
            jSuppression.append(tmpl.jSuppression)
            #del tmpl
            k +=1
            maxa = 100000
            if k>maxa:
                k=0
                self.templates = self.templates[maxa:]
            #if i % 100000 == 0:
             #   gc.collect()
        del self.templates
        gc.collect()
        print ('done collecting')
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
        col14= fits.Column(name="id_simulated_gal",format="K",array=p0)
        col15= fits.Column(name="id_simulated_PSF",format="K",array=p0_PSF)
        if 1==1:
        #try:
            print (np.array(class_))
            col16 = fits.Column(name="id_gal",format="K",array=id_gal)
            col17 = fits.Column(name="class",format="128A",array=np.array(class_))
        #except:
        #    pass
        if 1==1:
            
            cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17])
        #except:
        #    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15])
        tbhdu = fits.BinTableHDU.from_columns(cols, header=self.hdr)

        tbhdu.header['TIER_NUM'] = np.int32(TIER_NUM)
        tbhdu.header['EFFAREA'] = EFFAREA 
        print(tbhdu.header)
        prihdu = fits.PrimaryHDU()
        try:
            tbhdu.header['WT_SIG'] = tbhdu.header['WT_SIGMA']
        except:
            pass

     
        
        thdulist = fits.HDUList([prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
        return

    
    
    
    
    
    
def zero_padd_image(img,img_ref):
        '''
        it zero-padd the psf solutions such that they have the same size as the images
        '''
        
        size_x = img_ref.shape[0]
        size_y = img_ref.shape[1]
        mute = np.zeros((size_x,size_y))
        size_psf_x = img.shape[0]
        size_psf_y = img.shape[1]
        # sometimes the PiFF stamp is larger than the image
        if size_psf_x>size_x:
            dx = -np.int((size_x-size_psf_x)/2)
            mute = img[dx:dx+size_x,:][:,dx:dx+size_x]
                    # if the PiFF stamp is smaller than the image
        elif size_psf_x<size_x:
            dx = np.int((size_x-size_psf_x)/2)
            mute[dx:dx+size_psf_x,:][:,dx:dx+size_psf_x] = self.psf[index_band][exp]
        
        return mute


            
        
########################################################
#
# TILE SIMULATIONS SPECIFIC CODE!
#
########################################################

def _add_T_and_scale(obj_data, scale):
    add_dt = [('T', 'f4')]
    objs = eu.numpy_util.add_fields(obj_data, add_dt)
    objs['flux'] *= scale**2
    min_sigma = scale
    min_T = 2*min_sigma**2
    T = 2*objs['hlr']**2
    T = T.clip(min=min_T)
    objs['T'] = T
    return objs

            

    
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
        
        print ('SHAPE ', np.array(self.meb).shape,len(self.id))
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
            col.append(fits.Column(name="blend",format="D",array=self.is_it_a_blend))
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

            try:
                cov_ = np.array(self.cov).astype(np.float32)
                cov_[:,0] *= (1.+1./np.sqrt(self.len_v))**2
                
            except:
                cov_ = np.array(self.cov).astype(np.float32)
                print ('no COV[0,0] correction')
            #    
            col.append(fits.Column(name="covariance",format="15E",array=cov_))
        
        try:
            col.append(fits.Column(name="bkg",format="D",array=self.bkg))
            
            
            
        except:
            pass
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
            
        
        col.append(fits.Column(name="AREA",format="D",array=self.AREA))
        
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
        self.prihdu.header['STAMPS'] = stamp
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
      
        return

    
def get_config_meds():
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
    return config_meds
    
def get_box_sizes(cat):

    
    """
    get box sizes that are wither 2**N or 3*2**N, within
    the limits set by the user
    """
    
    
    config_meds = get_config_meds()

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
    config_meds = get_config_meds()
    FWHM_FAC = 2*np.sqrt(2*np.log(2))
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


def initialise_entries(tab_targets):

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
        tab_targets.is_it_a_blend = []
        tab_targets.bkg = []
        tab_targets.len_v = []
        tab_targets.des_id = []
        tab_targets.photoz = []
        return tab_targets
    
    
def setup_templates_table(config):

    params_template = {}
    params_template['n'] = config['n'] 
    params_template['sigma'] = config['sigma'] 


    mute_b = dict()
    for i in range(len(config['band_dict'])):
        mute_b[config['band_dict'][i][0]] = config['band_dict'][i][1]
    params_template['bands'] = config['bands']
    params_template['band_dict'] = dict()
    params_template['band_dict']['bands'] = list(config['bands'])
    w = []
    for b in config['bands']:
        w.append(mute_b[b])
    params_template['band_dict']['weights'] = list(w)
    params_template['band_dict']['index'] = list(np.arange(len(w)))





    params_template['bands'] = config['bands']
    tab = TemplateTable(n = config['n'],
               sigma = config['sigma'],
                sn_min = 0.,
                sigma_xy = 0.,
                sigma_flux = 0.,
                sigma_step = 0.,
                sigma_max = 0.,
                xy_max = 0.)

    tab_detections = DetectionsTable(params_template)

    return tab_detections,params_template
            
def set_noise(do_templates,config):
    '''
    it sets the noise of the tiles.
    This is controlled by the entries 'noise_ext_templates' or 'noise_ext' for templates or targets.
    if noise_ext is of the type [media,std,'GAUSS'], it will draw noise from a Gaussian distribution.
    if noise_ext is of the type [lower,upper], it ll randomly draw from the interval [lower,upper].
    if noiseext is of the type [value], noise will be fixed to 'value'.
    
    '''
    if do_templates:
        if len(config['noise_ext_templates'])==3:
            noise_ext = np.random.normal(config['noise_ext_templates'][0],config['noise_ext_templates'][1])
            if noise_ext<= 0:
                noise_ext = config['noise_ext_templates'][0]

        elif len(config['noise_ext_templates'])==2:
            noise_ext = config['noise_ext_templates'][0] + np.random.randint(0,10000,1)*0.0001*(config['noise_ext'][1]-config['noise_ext'][0])
        else:
            noise_ext = config['noise_ext_templates'][0]
    else:
        if len(config['noise_ext'])==3:
            noise_ext = np.random.normal(config['noise_ext'][0],config['noise_ext'][1])
            if noise_ext<= 0:
                noise_ext = config['noise_ext'][0]

        elif len(config['noise_ext'])==2:
            noise_ext = config['noise_ext'][0] + np.random.randint(0,10000,1)*0.0001*(config['noise_ext'][1]-config['noise_ext'][0])
        else:
            noise_ext = config['noise_ext'][0]
            
    return noise_ext



def select_obj(x,y,radius):
    new_DV = np.vstack([x,y]).T
    
    mask = np.array([True]*len(x))
    ipx_ref = np.arange(len(mask))
        
    
    r = len(mask[mask])
    redoit = True
    
    # this removes objects too close ++++++++++++
    while redoit:
        mask_w_dummy = copy.deepcopy(mask)
        
        YourTreeName = scipy.spatial.cKDTree(new_DV[mask_w_dummy], leafsize=100)
        
        d,ipx__ = YourTreeName.query(new_DV[mask_w_dummy], k=2, distance_upper_bound=radius)

        ipx_ = []
        for dd in ipx__:
            ipx_.append(np.sort(dd))
        ipx_ = np.array(ipx_)
        
        
        a = np.zeros((ipx_ref.shape[0]))
        ipx_ref_ = ipx_ref[mask_w_dummy]
        

        u_ = (ipx_[ipx_[:,1]<len(mask[mask_w_dummy])])
    
        for ii in frogress.bar(range(len(u_))):
            ipx = u_[ii]
            
            #print (ipx,a[ipx_ref[mask_w_dummy][ipx[0]]])
            
            if (a[ipx_ref_[ipx[0]]] ==0) and (a[ipx_ref_[ipx[1]]] == 0): 
                mask[ipx_ref[mask_w_dummy][ipx[1]]] = False
                # remove obj list
                a[ipx_ref_[ipx[0]]] = 1
                a[ipx_ref_[ipx[1]]] = 1

        if r == len(mask[mask]):
            redoit = False
        r = len(mask[mask])
   
    # sometimes, removed objects are too many. ++++++++++++
    #print (new_DV[mask,:])
    ipx_ref_ = ipx_ref[~mask_w_dummy]
    add_ = new_DV[~mask_w_dummy,:]
    removed_excess = 0

    for i in frogress.bar(range(add_.shape[0])):
        YourTreeName = scipy.spatial.cKDTree(new_DV[mask], leafsize=100) 
        d,ipx_ = YourTreeName.query(add_[i].reshape(1,-1), k=1, distance_upper_bound=10*radius)
  


        if d>radius:
            #print ('put back: ',x[ipx_ref_[i]])
            #print (x[ipx_ref_[i]])
            mask[ipx_ref_[i]] = True
            removed_excess +=1

            
    #print ('')
   # print('removed excess ',removed_excess)

    return mask
 
    
    
def subtract_background_(image, seg_, mask_):
    # excludes masked pixels and pixels where objects are detected
    seg = copy.deepcopy(seg_)
    segg0 = copy.deepcopy(seg_)
    for i in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
        seg += (np.roll(segg0, i, axis = 0) + np.roll(segg0, i, axis = 1))

    mask__ = copy.deepcopy(mask_)
    segg0 = copy.deepcopy(mask_)
    for i in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
        mask__ += (np.roll(segg0, i, axis = 0) + np.roll(segg0, i, axis = 1))

        
    mask = (seg == 0) & (mask__ == 0) 
    maskarr=np.ones(np.shape(mask),dtype='int')
    maskarr[~mask] = 0
    uu = 6
    maskarr[uu:-uu,uu:-uu]=0
    v = image[np.where(maskarr==1)]

    if len(v)>50: # at least 50 pixels...
        correction = np.median(v)
        return correction,len(v)
    else:
        
        correction = 0.
        return correction,0
    
                                                                
                    
def mfrac_(mask_,shape,wcs):                                                             
    psf_fwhm = 2.
    Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)

    psf_pars = [0.0, 0.0, 0.0, 0.00, Tpsf, 1.0]
    psf_gmix = ngmix.GMixModel(pars=psf_pars, model="turb")
    
    jac = jacobian=ngmix.jacobian.Jacobian(row=wcs.xy0[1],
                          col=wcs.xy0[0],
                          dudrow=wcs.jac[0,1],
                          dudcol=wcs.jac[0,0],
                          dvdrow=wcs.jac[1,1],
                          dvdcol=wcs.jac[1,0])
    test_filter = psf_gmix.make_image((shape),jac,fast_exp=True)
    mfrac = np.sum(test_filter.flatten()[mask_.flatten()!=0.])/np.sum(test_filter).flatten()
    return mfrac
def quick_mask_interp(image_sheared,mask,noise_rms):
    mask |= np.rot90(mask)
    import shredder
    import esutil as eu
    detection_cat, seg_map = sxdes.run_sep(image_sheared, noise_rms)
    shredder_cat = dict()
    seed = 113
    rng = np.random.RandomState(seed)
    guess_model = 'dev'
    wmp = np.ones((shape,shape))*1./noise_rms**2


    psf_o = ngmix.Observation(im_psf,jacobian=jac)

    mbobs_ = ngmix.Observation(image_sheared,
            weight=wmp,
            meta={"orig_row": (shape)/2., "orig_col": (shape)/2.},
            jacobian=jac,
            psf=psf_o)

    mbobs_ = ngmix.observation.get_mb_obs(mbobs_)

    dt = np.dtype([('shift', '<f4', (2,)),('flux', 'f4'), ('hlr', 'f4'),('col', 'f4'),('row', 'f4')])


    m = []
    for i in range(len(detection_cat['flux'])):
        m.append(((0.,0.),detection_cat['flux'][i],
    detection_cat['flux_radius'][i],
    detection_cat['x'][i],
    detection_cat['y'][i]))
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
    
    for i in range(len(models)):
        if i== 0:
            mod_tot = models[i]
        else:
            mod_tot += models[i]

    image_sheared[mask != 0 ] = mod_tot[mask != 0 ]
    noise = np.random.normal(size = (image_sheared.shape[0],image_sheared.shape[0]))*noise_rms
    image_sheared[mask != 0 ] += noise[mask != 0 ]
    return image_sheared
