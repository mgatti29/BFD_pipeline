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
import ngmixer
import ngmix.gmix as gmix
import gc
import ngmix


def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

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
                    self.imlist[b][i][mask] =- correction
                
                
    def compute_mfrac(self):
        '''
        It computes the fraction of pixels masked in each exposure. 
        '''

        self.size =  [[len(self.masklist[b][i].flatten()) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        self.mfrac = [[1.*len(np.zeros(self.size[b][i])[(self.wtlist[b][i].flatten()==0.) & (self.masklist[b][i].flatten()!=0.)])/self.size[b][i] for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
    
    def check_mfrac(self, limit=0.1, use_COADD_only = False):
        '''
        It checks which bands have exposures with a fraction of pixels masked above the limit
        and return False for the bands with masked pixels
        '''
        self.compute_mfrac()
        bands_not_masked = dict()
        for b in range(len(self.bands)):
            bands_not_masked[self.bands[b]] = True
            if use_COADD_only:
                if self.mfrac[b][0] > limit:
                    bands_not_masked[self.bands[b]] = False
            else:
                for i in range((self.ncutout[b])):
                    if self.mfrac[b][i] > limit:
                        bands_not_masked[self.bands[b]] = False
                  
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
            
    def compute_noise(self, del_list = True):
        self.noise_rms =  [[np.mean(1./self.wtlist[b][i][(self.wtlist[b][i]!= 0.) & (self.masklist[b][i] == 0)]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
        if del_list:
            self.wtlist = None
            self.masklist = None
            gc.collect()
                
            
    def discard_exposures_mfrac(self,limit=0.1):
        '''
        It drops exposures with fraction of pixels masked > limit.
        '''
        # It computes the fraction of pixels masked in each exposure
        self.compute_mfrac()
        remove_list =[]
        self.exposures_discarded = 0
        self.tot_exposures = 0
        
        # it makes a list of masked exposures
        for index_band in range(self.n_bands):
            for exp in range(self.ncutout[index_band])[::-1]:
                self.tot_exposures+=1
                if self.mfrac[index_band][exp]>limit:
                    remove_list.append([index_band,exp])
                    self.exposures_discarded+=1
                    
        # it drops the selected exposures
        for (index_band,exp) in remove_list:
            self.ncutout[index_band]-=1
            del self.imlist[index_band][exp]
            del self.seglist[index_band][exp]
            del self.wtlist[index_band][exp]
            del self.masklist[index_band][exp]
            del self.size[index_band][exp]
            del self.mfrac[index_band][exp]
            try:
                del self.psf[index_band][exp]
            except:
                pass
            
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
            

    def compute_moments_multiband_varying_sigma(self, use_COADD_only = False, bands = ['r', 'i', 'z'], 
                                  band_dict = {'r':bfd.BandInfo(0.5,0),'i':bfd.BandInfo(0.5,1),'z':bfd.BandInfo(0.5,2)},plot = True, MOF_subtraction = False):
        
        imgs = []
        wcss = []
        psfs = []
        noise = []
        bandlist = []
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
                    try:
                        noise_rms = self.noise_rms[index_band][exp]
                    except:
                        noise_rms = np.mean(1./self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])
                    noise.append(noise_rms)
                    bandlist.append(self.bands[index_band])
        kds = bfd.multiImage(imgs, (0,0), psfs, wcss, pixel_noiselist = noise, bandlist = bandlist)


        steps = 50
        ss = np.linspace(0,4,steps)
        sig = np.zeros((steps,5))
        nn = np.zeros((steps,5))
        for i,s in enumerate(ss):
            wt = mc.KSigmaWeight(sigma = s) 
            mm = bfd.MultiMomentCalculator(kds, wt, band_dict = band_dict)
            mm.recenter()
            mom = mm.get_moment(0.,0.)
            sig[i,0] = mom.even[mom.M0]
            sig[i,1] = mom.even[mom.M1]
            sig[i,2] = mom.even[mom.M2]
            sig[i,3] = mom.even[mom.MR]
            sig[i,4] = mom.even[mom.MC]
            nn[i,0] = mm.get_covariance()[0][mom.M0,mom.M0]
            nn[i,1] = mm.get_covariance()[0][mom.M1,mom.M1]
            nn[i,2] = mm.get_covariance()[0][mom.M2,mom.M2]
            nn[i,3] = mm.get_covariance()[0][mom.MR,mom.MR]
            nn[i,4] = mm.get_covariance()[0][mom.MC,mom.MC]


        if plot: 
            fig, ax = plt.subplots(2, 5,figsize=(15,6))
            ax[0,0].plot(ss, sig[:,0]/np.sqrt(nn[:,0]),label='M0')
            ax[0,1].plot(ss, sig[:,1]/np.sqrt(nn[:,1]),label='M1')
            ax[0,2].plot(ss, sig[:,2]/np.sqrt(nn[:,2]),label='M2')
            ax[0,3].plot(ss, sig[:,3]/np.sqrt(nn[:,3]),label='MR')
            ax[0,4].plot(ss, sig[:,4]/np.sqrt(nn[:,4]),label='MC')

            ax[0,0].set_xlabel('weight sigma')
            ax[0,1].set_xlabel('weight sigma')
            ax[0,2].set_xlabel('weight sigma')
            ax[0,3].set_xlabel('weight sigma')
            ax[0,4].set_xlabel('weight sigma')
            ax[1,0].set_xlabel('weight sigma')
            ax[1,1].set_xlabel('weight sigma')
            ax[1,2].set_xlabel('weight sigma')
            ax[1,3].set_xlabel('weight sigma')
            ax[1,4].set_xlabel('weight sigma')

            ax[0,0].set_ylabel('S/N Mf')
            ax[0,1].set_ylabel('S/N M1')
            ax[0,2].set_ylabel('S/N M2')
            ax[0,3].set_ylabel('S/N Mr')
            ax[0,4].set_ylabel('S/N Mc')

            ax[1,0].set_ylabel('Mf')
            ax[1,1].set_ylabel('M1')
            ax[1,2].set_ylabel('M2')
            ax[1,3].set_ylabel('Mr')
            ax[1,4].set_ylabel('Mc')

            ax[0,0].grid()

            ax[1,0].plot(ss, sig[:,0], 'r-',label='M0')
            ax[1,1].plot(ss, sig[:,1], 'r-',label='M0')
            ax[1,2].plot(ss, sig[:,2], 'r-',label='M0')
            ax[1,3].plot(ss, sig[:,3], 'r-',label='M0')
            ax[1,4].plot(ss, sig[:,4], 'r-',label='M0')
            plt.tight_layout()
            plt.show()

        


    def compute_moments_multiband(self, sigma = 3, use_COADD_only = False, bands = ['r', 'i', 'z'], 
                                  band_dict = {'r':bfd.BandInfo(0.5,0),'i':bfd.BandInfo(0.5,1),'z':bfd.BandInfo(0.5,2)}, MOF_subtraction = False):
        '''
        Compute moments combining exposures and bands
        '''
        
        imgs = []
        wcss = []
        psfs = []
        noise = []
        bandlist = []
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
                    try:
                        noise_rms = self.noise_rms[index_band][exp]
                    except:
                        noise_rms = np.mean(1./self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])
                    noise.append(noise_rms)
                    bandlist.append(self.bands[index_band])
        
          
        kds = bfd.multiImage(imgs, (0,0), psfs, wcss, pixel_noiselist = noise, bandlist = bandlist)
        wt = mc.KSigmaWeight(sigma = sigma) 
        mul = bfd.MultiMomentCalculator(kds, wt, band_dict = band_dict)
        self.xyshift, error,msg = mul.recenter()
        self.moments = mul
        

        
        return mul
    
    def compute_moments_varying_sigma(self, exp = 0, band = 'i', plot = True, MOF_subtraction = False):
        '''
        computes moments varying sigma
        '''
        self.mmlist = []
        
        index_band = np.arange(len(self.bands))[np.array(np.in1d(self.bands,band))]
        if len(index_band) == 0:
            pass
        else:
            index_band = index_band[0]
            N = self.imlist[index_band][exp].shape[0]
            try:
                noise_rms = self.noise_rms[index_band][exp]
            except:
                noise_rms = np.mean(1./self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])
            
            if MOF_subtraction:
                img = self.imlist[index_band][exp] - self.MOF_model_rendered[index_band][exp]
            else:
                img = self.imlist[index_band][exp]
                    
            kdata = mc.simpleImage(img, (0,0), self.psf[index_band][exp], pixel_noise=noise_rms, wcs = self.wcslist[index_band][exp])

            steps = 50
            ss = np.linspace(0,4,steps)
            sig = np.zeros((steps,5))
            nn = np.zeros((steps,5))

            for i,s in enumerate(ss):
                wt = mc.KSigmaWeight(sigma=s)
                mm = mc.MomentCalculator(kdata,wt)
                mm.recenter() #in arcsec
                mom = mm.get_moment(0.,0.)
                sig[i,0] = mom.even[mom.M0]
                sig[i,1] = mom.even[mom.M1]
                sig[i,2] = mom.even[mom.M2]
                sig[i,3] = mom.even[mom.MR]
                sig[i,4] = mom.even[mom.MC]
                nn[i,0] = mm.get_covariance()[0][mom.M0,mom.M0]
                nn[i,1] = mm.get_covariance()[0][mom.M1,mom.M1]
                nn[i,2] = mm.get_covariance()[0][mom.M2,mom.M2]
                nn[i,3] = mm.get_covariance()[0][mom.MR,mom.MR]
                nn[i,4] = mm.get_covariance()[0][mom.MC,mom.MC]


            if plot: 
                fig, ax = plt.subplots(2, 5,figsize=(15,6))
                ax[0,0].plot(ss, sig[:,0]/np.sqrt(nn[:,0]),label='M0')
                ax[0,1].plot(ss, sig[:,1]/np.sqrt(nn[:,1]),label='M1')
                ax[0,2].plot(ss, sig[:,2]/np.sqrt(nn[:,2]),label='M2')
                ax[0,3].plot(ss, sig[:,3]/np.sqrt(nn[:,3]),label='MR')
                ax[0,4].plot(ss, sig[:,4]/np.sqrt(nn[:,4]),label='MC')

                ax[0,0].set_xlabel('weight sigma')
                ax[0,1].set_xlabel('weight sigma')
                ax[0,2].set_xlabel('weight sigma')
                ax[0,3].set_xlabel('weight sigma')
                ax[0,4].set_xlabel('weight sigma')
                ax[1,0].set_xlabel('weight sigma')
                ax[1,1].set_xlabel('weight sigma')
                ax[1,2].set_xlabel('weight sigma')
                ax[1,3].set_xlabel('weight sigma')
                ax[1,4].set_xlabel('weight sigma')

                ax[0,0].set_ylabel('S/N Mf')
                ax[0,1].set_ylabel('S/N M1')
                ax[0,2].set_ylabel('S/N M2')
                ax[0,3].set_ylabel('S/N Mr')
                ax[0,4].set_ylabel('S/N Mc')

                ax[1,0].set_ylabel('Mf')
                ax[1,1].set_ylabel('M1')
                ax[1,2].set_ylabel('M2')
                ax[1,3].set_ylabel('Mr')
                ax[1,4].set_ylabel('Mc')

                ax[0,0].grid()

                ax[1,0].plot(ss, sig[:,0], 'r-',label='M0')
                ax[1,1].plot(ss, sig[:,1], 'r-',label='M0')
                ax[1,2].plot(ss, sig[:,2], 'r-',label='M0')
                ax[1,3].plot(ss, sig[:,3], 'r-',label='M0')
                ax[1,4].plot(ss, sig[:,4], 'r-',label='M0')
                plt.tight_layout()
                plt.show()
                
    def compute_psf_fwhm(self,use_COADD_only = False):
        self.psf_fwhm = []
        for index_band in range(self.n_bands):
            mute_bands =[]
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[index_band]
            for exp in range(start, end):   
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
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[index_band]
            for exp in range(start, end):    
                # it defines the jacobian in the ngmix format.
                jac = self.jaclist[index_band][exp]
                jac['row'] = jac['row0']
                jac['col'] = jac['col0']

                min_linear_scale = np.min([jac['dudcol'],jac['dvdrow']])
                jacobian_ngmix = ngmix.Jacobian(**jac)

                # it converts the psf image into a ngmi observation
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
            self.psf_params.append(mute_list)
            

            
        self.psf_params_average = {'g1': 0. ,'g2': 0., 'T': 0.}
        bands_to_use = self.bands_not_masked
        w = 0.
        for bx, band in enumerate(band_dict):
            index_band = np.arange(len(self.bands))[np.array(np.in1d(self.bands,band))][0]
            if band in bands_to_use:
                w+=band_dict[band].weight
                self.psf_params_average['g1'] += band_dict[band].weight * self.psf_params[index_band][0]['g1']
                self.psf_params_average['g2'] += band_dict[band].weight * self.psf_params[index_band][0]['g2']
                self.psf_params_average['T']  += band_dict[band].weight * self.psf_params[index_band][0]['T'] 

        self.psf_params_average['g1'] = self.psf_params_average['g1']/w
        self.psf_params_average['g2'] = self.psf_params_average['g2']/w
        self.psf_params_average['T']  = self.psf_params_average['T']/w
        
    

import pyfits as pf
class MOF_table:
    def __init__(self, path):
        '''
        Loads into memory a MOF catalog.
        MOF has 3 entries - 1 table is for the 
        '''
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
                    
#ID = 712144440



def render_gal(gal_pars,psf_pars,wcs,shape,model='bdf',debug=False, g1 = None, g2 = None):
    psf_gmix=gmix.GMix(pars=psf_pars)
    e1,e2,T=psf_gmix.get_e1e2T()

    det=np.abs(wcs.getdet()) 
    jac=gmix.Jacobian(row=wcs.xy0[1],
                      col=wcs.xy0[0],
                      dudrow=wcs.jac[0,1],
                      dudcol=wcs.jac[0,0],
                      dvdrow=wcs.jac[1,1],
                      dvdcol=wcs.jac[1,0])

    if model=='bdf':
        gmix_sky = gmix.GMixBDF(gal_pars)
    elif model=='cm':
        # fracdev index is 6
        # tdbyte index is 7
        # normal pars are 0-5
        gmix_sky = gmix.GMixCM(gal_pars[6],gal_pars[7],gal_pars[0:6])
    else:
        raise Exception("must supply valid model (bdf or cm)")

    if (g1  != None) and (g2  != None):
        gmix_sky = gmix_sky.get_sheared(g1,g2)

    gmix_image = gmix_sky.convolve(psf_gmix)
    try:
        image = gmix_image.make_image((shape,shape), jacobian=jac, fast_exp=True)
    except:
        image=np.zeros((shape,shape))

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
        
    def compute_moments(self, sigma, bands = 'All', use_COADD_only = True, flags = 'All', MOF_subtraction = True, band_dict = {'r':bfd.BandInfo(0.5,0),'i':bfd.BandInfo(0.5,1),'z':bfd.BandInfo(0.5,2)}, chunk_range = None):
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
                try:
                
                    band_dict_to_use = dict()
                    for bx in bands_to_use:
                        band_dict_to_use[bx]  = band_dict[bx]
       
                    self.images[i].compute_moments_multiband(sigma = sigma, bands = bands_to_use, use_COADD_only = use_COADD_only, MOF_subtraction = MOF_subtraction, band_dict = band_dict_to_use)
                    
                    # erase info image:
                    self.images[i].imlist = None
                    self.images[i].seglist = None
                    self.images[i].masklist = None
                    self.images[i].wtlist = None
                    self.images[i].psf = None
                    gc.collect()
                except:
                    print ('failed moments computation; object ',i, bands_to_use,self.images[i].bands_not_masked,self.images[i].bands)
                    self.images[i].imlist = None
                    self.images[i].seglist = None
                    self.images[i].masklist = None
                    self.images[i].wtlist = None
                    self.images[i].psf = None
                    gc.collect()
            else:
                #erase info image
                self.images[i].imlist = None
                self.images[i].seglist = None
                self.images[i].masklist = None
                self.images[i].wtlist = None
                self.images[i].psf = None
                gc.collect()
                
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
                
    
    def make_templates(self, SN_limit = 0, flags = 'All', make_only_one = False):
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
                    if make_only_one:
                        t = make_one_template(self.images[i].moments,self.params['sigma_xy'],sigma_flux=self.params['sigma_flux'], sn_min=self.params['sn_min'], sigma_max=self.params['sigma_max'], 
                    sigma_step=self.params['sigma_step'], xy_max=self.params['xy_max'],image_id = self.images[i].image_ID) 
                    else:
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
        
        
def make_one_template(self, sigma_xy, sigma_flux=1., sn_min=0., sigma_max=6.5, sigma_step=1., xy_max=2.,
                           image_id =0, **kwargs):
        ''' Return a list of Template instances that move the object on a grid of
        coordinate origins that keep chisq contribution of flux and center below
        the allowed max.
        sigma_xy    Measurement error on target x & y moments (assumed equal, diagonal)
        sigma_flux  Measurement error on target flux moment
        sn_min      S/N for minimum flux cut applied to targets
        sigma_max   Maximum number of std deviations away from target that template will be used
        sigma_step  Max spacing between shifted templates, in units of measurement sigmas
        xy_max      Max allowed centroid shift, in sky units (prevents runaways)
        '''
        xyshift, error, msg = self.recenter()
        if error:
            return None, "Center wandered too far from starting guess or failed to converge"
        # Determine derivatives of 1st moments on 2 principal axes,
        # and steps will be taken along these grid axes.
        jacobian0 = self.xy_jacobian(np.zeros(2))
        eval, evec = np.linalg.eigh(jacobian0)
        if np.any(eval>=0.):
            return None, "Template galaxy center is not at a flux maximum"

        detj0 = np.linalg.det(jacobian0) # Determinant of Jacobian
        xy_step = np.abs(sigma_step * sigma_xy / eval)
        da = xy_step[0] * xy_step[1]

        # Offset the xy grid by random phase in the grid
        xy_offset = np.random.random(2) - 0.5

        # Now explore contiguous region of xy grid that yields useful templates.
        result = []
        grid_try = set( ( (0,0),) )  # Set of all grid points remaining to try
        grid_done = set()           # Grid points already investigated

        flux_min = sn_min * sigma_flux
        
        while len(grid_try)>0:
            # Try a new grid point
            mn = grid_try.pop()

            grid_done.add(mn)  # Mark it as already attempted
            xy = np.dot(evec, xy_step*(np.array(mn) + xy_offset))  # Offset and scale
            # Ignore if we have wandered too far
            if np.dot(xy,xy) > xy_max*xy_max:
                continue
            m = self.get_moment(xy[0], xy[1])
            e = m.even
            detj = 0.25 * ( e[m.MR]**2-e[m.M1]**2 - e[m.M2]**2)
            # Ignore if determinant of Jacobian has gone negative, meaning
            # we have crossed out of convex region for flux
            if detj <= 0.:
                continue

            # Accumulate chisq that this template would have for a target
            # First: any target will have zero MX, MY
            chisq = (m.odd[m.MX]**2 + m.odd[m.MY]**2) / sigma_xy**2
            # Second: there is suppression by jacobian of determinant
            chisq += -2. * np.log(detj/detj0)
            # Third: target flux will never be below flux_min
            if (e[m.M0] < flux_min):
                chisq += ((flux_min -e[m.M0])/sigma_flux)**2
            if chisq <= sigma_max*sigma_max:
                # This is a useful template!  Add it to output list
                tmpl = self.get_template(xy[0],xy[1])
                tmpl.id = image_id[0]
                tmpl.nda = tmpl.nda * da
                tmpl.jSuppression = detj / detj0
                result.append(tmpl)
                # Try all neighboring grid points not yet tried

        if len(result)==0:
            result.append(None)
            result.append("no templates made")
        return result
                