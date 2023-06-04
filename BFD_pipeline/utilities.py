import pickle
import glob
import frogress
import astropy.io.fits as fits
from bfd.keywords import *
import os
import gc
import numpy as np


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
    """
    It takes as a input a an array of moments (Mf,Mr,M1,M2,Mc) and returns the angle described by M1 and M2.
    
    Returns
    -------
    flux moment SN
    """
    return np.arctan((e[3]/e[2]))


def produce_cov(t):
    """
    It takes as a input a loaded fits file describing the targets and return the flux moment covariance
    
    Returns
    -------
    flux moment covariance
    """
    return (t[1].data['covariance'][:,0])

def produce_mf(t):
    """
    It takes as a input a loaded fits file describing the targets and return the flux moment
    
    Returns
    -------
    flux moment
    """
    return t[1].data['moments'][:,0]

def produce_sn(t):    
    """
    It takes as a input a loaded fits file describing the targets and return the SN of the flux moment.
    
    Returns
    -------
    flux moment SN
    """
    return t[1].data['moments'][:,0]/np.sqrt(t[1].data['covariance'][:,0])

def collapse(path, name_):
    """
    Function to collapse data from multiple files into a single FITS file.
    This is mostly used to collapse templates files together after measuring their moments in chunks.
    
    Returns
    -------
    None
    """
    # Append the 'name_' value to the 'path'
    path = path + name_
    
    # Get a list of files in the specified path
    files = glob.glob(path + '*')
    
    # Iterate over the files
    for ii in frogress.bar(range(len(files))):
        # Get the current file
        file_ = files[ii]
        
        try:
            # Try to open the file using the FITS module
            mute = fits.open(file_)
        except:
            # If an exception occurs, move to the next file
            pass
        
        if ii == 0:
            # If it is the first file, initialize the 'mf' (moment flux) array with the moments data
            mf = mute[1].data['moments'][:, 0]
        else:
            # If it is not the first file, horizontally stack the moments data to 'mf'
            mf = np.hstack([mf, mute[1].data['moments'][:, 0]])

    # Create a new FITS HDU list with a primary HDU
    hdulist = fits.HDUList([fits.PrimaryHDU()])
    cols = []
    
    # Append a new column named "notes" with the appropriate format and array to 'cols'
    cols.append(fits.Column(name="notes", format="K", array=0*np.ones_like(mf)))
    new_cols = fits.ColDefs(cols)
    
    # Create a new binary table HDU from the existing columns and add it to the HDU list
    hdu = fits.BinTableHDU.from_columns(mute[1].columns + new_cols)
    
    # Copy header keys related to weightN and weightSigma from the first file to the new HDU's header
    for key in (hdrkeys['weightN'], hdrkeys['weightSigma']):
        hdu.header[key] = mute[0].header[key]
    
    for cname in mute[1].columns.names:
        sofar = 0
        for ii, file_ in enumerate(files):
            mute = fits.open(file_)
            nn = len(mute[1].data['moments'][:, 0])
            # Copy the data from each file to the new HDU's corresponding column
            hdu.data[cname][sofar:sofar+nn] = mute[1].data[cname]
            sofar += nn

    # Copy header keys related to weightN and weightSigma from the first file to the primary HDU's header
    for key in (hdrkeys['weightN'], hdrkeys['weightSigma']):
        hdulist[0].header[key] = mute[0].header[key]
    
    # Copy the 'STAMPS' key from the first file's header to the primary HDU's header
    hdulist[0].header['STAMPS'] = mute[0].header['STAMPS']
    
    # Append the new HDU to the HDU list and delete the HDU to free memory
    hdulist.append(hdu)
    del hdu

    # Remove the files that have been collapsed
    for file_ in files:
        try:
            os.remove(file_)
        except:
            pass

    try:
        # Try to write the HDU list to a FITS file with the specified path
        hdulist.writeto(path + '.fits')
    except:
        try:
            hdulist.writeto(path+'.fits',clobber = True)# 
        except:
            pass

    
    '''
    The next bit is not supported - it collapses high SN  targets that were saved to be used as templates.
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
    '''
    
def load_obj(name):
    # Function to load an object from a file using pickle module
    try:
        # Try to open the file with the given name and '.pkl' extension in read binary mode
        with open(name + '.pkl', 'rb') as f:
            # Use pickle.load() to deserialize and load the object from the file 'f'
            # Return the loaded object
            return pickle.load(f)
    except:
        # If an exception occurs (possibly due to encoding issues), try loading the object with 'latin1' encoding
        with open(name + '.pkl', 'rb') as f:
            # Use pickle.load() with 'latin1' encoding to deserialize and load the object from the file 'f'
            # Return the loaded object
            return pickle.load(f, encoding='latin1')




def save_moments_targets(self,fitsname,config):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        
        # The code will save different columns into a fits file depending on they exist.

        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        col.append(fits.Column(name="moments",format="5E",array=self.moment))
        col.append(fits.Column(name="Mf_per_band",format="{0}E".format(np.array(self.meb).shape[1]),array=self.meb))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))    
        col.append(fits.Column(name="ra",format="D",array=self.ra))
        col.append(fits.Column(name="dec",format="D",array=self.dec))
        PSF_moments = np.vstack([np.array(self.psf_Mf),np.array(self.psf_Mr),np.array(self.psf_M1),np.array(self.psf_M2)]).T
        col.append(fits.Column(name="psf_moments",format="4E",array=PSF_moments))
        l = np.array(self.orig_row).shape[1]
        col.append(fits.Column(name="DESDM_coadd_x",format="E",array=np.array(self.DESDM_coadd_x)[:,0]))
        col.append(fits.Column(name="DESDM_coadd_y",format="E",array=np.array(self.DESDM_coadd_y)[:,0]))
        col.append(fits.Column(name="orig_col",format="{0}E".format(l),array=np.array(self.orig_col)))
        col.append(fits.Column(name="orig_row",format="{0}E".format(l),array=np.array(self.orig_row)))
        col.append(fits.Column(name="ccd_number",format="{0}K".format(l),array=np.array(self.ccd_name)))
        col.append(fits.Column(name="bkg",format="D",array=self.bkg))
             
        try:
            if len(self.p0)>0:
                col.append(fits.Column(name="id_simulated_gal",format="K",array=self.p0))
                col.append(fits.Column(name="id_simulated_PSF",format="K",array=self.p0_PSF))
        except:
            pass
        
        try:
            if len(self.true_fluxes)>0:
                col.append(fits.Column(name="true_fluxes",format="{0}E".format(np.array(self.meb).shape[1]),array=self.true_fluxes))
        except:
            pass

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
        
                                              
        try:
            if len(self.photoz)>0:
                col.append(fits.Column(name="des_id",format="D",array=self.des_id))
                col.append(fits.Column(name="photoz",format="D",array=self.photoz))
        except:
            pass
        try:
            col.append(fits.Column(name="lenv",format="L",array=(self.len_v).astype(np.int)))
        except:
            pass
        if len(self.num_exp) == len(self.id):
            col.append(fits.Column(name="num_exp",format="K",array=self.num_exp))
        col.append(fits.Column(name="covariance",format="15E",array=np.array(self.cov).astype(np.float32)))
    
        if config['setup_image_sims']:
            self.prihdu.header['STAMPS'] = 1  
            col.append(fits.Column(name="AREA",format="K",array=np.zeros(len(self.id))))
        
        else:
            self.prihdu.header['STAMPS'] = 0
            col.append(fits.Column(name="AREA",format="K",array=self.AREA))
            

        # save to a fits file
        cols=fits.ColDefs(col)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)

        return

def save_obj(name, obj):
    # Function to save an object to a file using pickle module
    # Open a file with the given name and '.pkl' extension in write binary mode
    with open(name + '.pkl', 'wb') as f:
        # Use pickle.dump() to serialize and save the 'obj' to the file 'f'
        # Protocol 3 is used for Python 3.x compatibility
        pickle.dump(obj, f, protocol=3)



