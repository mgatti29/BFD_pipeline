import pickle
import glob
import frogress
import astropy.io.fits as fits
from bfd.keywords import *
import os
import gc
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=3)
        
def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')




# saving utilities
import astropy.io.fits as fits
import numpy as np
def save_moments_targets(self,fitsname,config):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        try:
            if len(self.p0)>0:
                col.append(fits.Column(name="id_simulated_gal",format="K",array=self.p0))
                col.append(fits.Column(name="id_simulated_PSF",format="K",array=self.p0_PSF))
        except:
            pass
        
        col.append(fits.Column(name="Mf_per_band",format="{0}E".format(np.array(self.meb).shape[1]),array=self.meb))
        try:
            if len(self.true_fluxes)>0:
                col.append(fits.Column(name="true_fluxes",format="{0}E".format(np.array(self.meb).shape[1]),array=self.true_fluxes))
        except:
            pass
        
        col.append(fits.Column(name="moments",format="5E",array=self.moment))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))
        
        col.append(fits.Column(name="ra",format="D",array=self.ra))
        col.append(fits.Column(name="dec",format="D",array=self.dec))
        
        PSF_moments = np.vstack([np.array(self.psf_Mf),np.array(self.psf_Mr),np.array(self.psf_M1),np.array(self.psf_M2)]).T
        col.append(fits.Column(name="psf_moments",format="4E",array=PSF_moments))

        try:
            if len(self.band1)>0:
                col.append(fits.Column(name="w_i",format="D",array=self.band1))
                col.append(fits.Column(name="w_r",format="D",array=self.band2))
                col.append(fits.Column(name="w_z",format="D",array=self.band3))
        except:
            pass
        try:
            l = np.array(self.meb).shape[1]
            col.append(fits.Column(name="cov_Mf_per_band",format="{0}E".format(l),array=np.array(self.cov_even_per_band)))
        except:
            pass
        
        
       # try:
        l = np.array(self.orig_row).shape[1]
        col.append(fits.Column(name="DESDM_coadd_x",format="E",array=np.array(self.DESDM_coadd_x)[:,0]))
        col.append(fits.Column(name="DESDM_coadd_y",format="E",array=np.array(self.DESDM_coadd_y)[:,0]))
        col.append(fits.Column(name="orig_col",format="{0}E".format(l),array=np.array(self.orig_col)))
        col.append(fits.Column(name="orig_row",format="{0}E".format(l),array=np.array(self.orig_row)))
        col.append(fits.Column(name="ccd_number",format="{0}K".format(l),array=np.array(self.ccd_name)))
        #except:
       #     pass
    
        #print (np.array(self.DESDM_coadd_x)[0,:])
        #print (np.array(self.DESDM_coadd_y).shape)
        #print (np.array(self.orig_col).shape)
        #print (np.array(self.orig_row).shape)
        #print (np.array(self.ccd_name).shape)
      

                                        
                                        
        
        try:
            if len(self.photoz)>0:
                col.append(fits.Column(name="des_id",format="D",array=self.des_id))
                col.append(fits.Column(name="photoz",format="D",array=self.photoz))
        except:
            pass

        #try:
        col.append(fits.Column(name="bkg",format="D",array=self.bkg))
        #except:
        #    pass
        try:
            col.append(fits.Column(name="lenv",format="L",array=(self.len_v).astype(np.int)))
        except:
            pass
        if len(self.num_exp) == len(self.id):
            col.append(fits.Column(name="num_exp",format="K",array=self.num_exp))
        col.append(fits.Column(name="covariance",format="15E",array=np.array(self.cov).astype(np.float32)))
    
        if config['setup_image_sims']:
            self.prihdu.header['STAMPS'] = 1  # Update value
            col.append(fits.Column(name="AREA",format="K",array=np.zeros(len(self.id))))
        
        else:
            self.prihdu.header['STAMPS'] = 0
            col.append(fits.Column(name="AREA",format="K",array=self.AREA))
            



        cols=fits.ColDefs(col)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
        return


# cllpase

def collapse(path,uuu):
    path = path+uuu
    files = glob.glob(path+'*')
    for ii in frogress.bar(range(len(files))):
        file_ = files[ii]
        try:
            mute = fits.open(file_)
        except:
            pass
        if ii ==0:
            mf  = mute[1].data['moments'][:,0]
        else:
            mf = np.hstack([mf,mute[1].data['moments'][:,0]])

    hdulist = fits.HDUList([fits.PrimaryHDU()])
    cols = []        
    cols.append(fits.Column(name="notes",format="K",array=0*np.ones_like(mf)))#noisetier[mask]*np.ones_like(noisetier[mask])))
    new_cols = fits.ColDefs(cols)
    hdu = fits.BinTableHDU.from_columns(mute[1].columns + new_cols)
    for key in (hdrkeys['weightN'],hdrkeys['weightSigma']):
        hdu.header[key] = mute[0].header[key]
    for cname in mute[1].columns.names:
        sofar = 0
        for ii, file_ in enumerate(files):
            mute = fits.open(file_)
            nn = len(mute[1].data['moments'][:,0])
            hdu.data[cname][sofar:sofar+nn] = mute[1].data[cname]
            sofar += nn  




    for key in (hdrkeys['weightN'],hdrkeys['weightSigma']):
        hdulist[0].header[key] = mute[0].header[key]
    hdulist[0].header['STAMPS'] = mute[0].header['STAMPS']
    hdulist.append(hdu)
    del hdu

    for file_ in files:
        try:
            os.remove(file_)
        except:
            pass

    try:
        hdulist.writeto(path+'.fits')
    except:
        try:
            hdulist.writeto(path+'.fits',clobber = True)# 
        except:
            pass
        
    # try with the templates as well ----
    
    files = glob.glob(path.split('targets/')[0]+'/targets/high_SN_templates_chunk_*{0}*'.format(uuu))
    if len(files)>0:
        save__ = dict()
        for file in files:
            tub = load_obj(file.split('.pkl')[0])
            for index in tub.keys():
                #try:
                 
                    save__[index] = dict()
                    save__[index]['moments'] = tub[index]['moments']
                    save__[index]['SN'] = tub[index]['SN']
                    save__[index]['index'] = tub[index]['index']
                    save__[index]['ra'] = tub[index]['ra']
                    save__[index]['dec'] = tub[index]['dec']
                #except:
                #    pass
            del tub
            gc.collect()
            os.remove(file)
        save_obj(path.split('targets/')[0]+'/targets/high_SN_templates_{0}'.format(uuu),save__)
    
    
  
