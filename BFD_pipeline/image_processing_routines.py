import astropy.io.fits as fits
import numpy as np
import pandas as pd
import galsim
import ngmix
import ngmix.gmix as gmix
import bfd
import copy
import bfd
from bfd import momentcalc as mc
from bfd.momentcalc import MomentCovariance
from bfd.keywords import *
import glob        
import os
import timeit



def grid_search(resolution):
        # Generate all possible combinations of r, i, z with the given resolution
        r = np.arange(0, 1 + resolution, resolution)
        i = np.arange(0, 1 + resolution, resolution)
        z = np.arange(0, 1 + resolution, resolution)

        valid_combinations = []

        for r_val in r:
            for i_val in i:
                for z_val in z:
                    if z_val:
                        if np.isclose(r_val + i_val + z_val, 1.0):
                            if z_val<0.3:
                                if i_val<0.8:
                                    if r_val<0.8:
                                        valid_combinations.append((int(r_val*1000)/1000.,int(i_val*1000)/1000.,int(z_val*1000)/1000.))

        return valid_combinations
    

def collapse(path, output_label):
    '''
    Processes and combines data from multiple FITS files located in a given directory,
    and saves the combined data into a single FITS file.

    Parameters:
    path (str): Path to the directory containing FITS files to be processed.
    output_label (str): Label for the output file.

    The function performs the following steps:
    1. Reads all FITS files in the specified directory.
    2. Concatenates data from a specific field across all files.
    3. Creates a new FITS file structure and populates it with aggregated data.
    4. Copies specific header information from the original files to the new file.
    5. Attempts to delete the original files.
    6. Writes the new aggregated data into a FITS file on disk.
    '''

    files = glob.glob(path + '*')

    # Initialize variables for aggregated data and header information
    all_data = []
    all_headers = []

    # Iterating over files, opening and processing each
    for ii, file_ in enumerate(files):
        with fits.open(file_) as mute:
            all_data.append(mute[1].data)
            all_headers.append(mute[0].header)
            # Aggregate 'moments' data from the first file
            if ii == 0:
                mf = mute[1].data['moments'][:, 0]
            else:
                # Concatenate 'moments' data from subsequent files
                mf = np.hstack([mf, mute[1].data['moments'][:, 0]])

    # Creating a new FITS file structure
    hdulist = fits.HDUList([fits.PrimaryHDU()])
    cols = []        
    cols.append(fits.Column(name="notes",format="K",array=0*np.ones_like(mf)))#noisetier[mask]*np.ones_like(noisetier[mask])))
    new_cols = fits.ColDefs(cols)

    # Assuming that the structure of all FITS files is the same,
    # using the columns from the first file for the BinTableHDU
    hdu = fits.BinTableHDU.from_columns(all_data[0].columns+new_cols)

    # Aggregating data from each column across all files
    for cname in all_data[0].columns.names:
        sofar = 0
        for data in all_data:
            nn = len(data[cname])
            hdu.data[cname][sofar:sofar + nn] = data[cname]
            sofar += nn

    # Additional header information
    for key in (hdrkeys['weightN'], hdrkeys['weightSigma']):
        hdulist[0].header[key] = mute[0].header[key]
    hdulist[0].header['STAMPS'] = mute[0].header['STAMPS']

    # Append the aggregated data to the HDU list
    hdulist.append(hdu)
    del hdu
    # Attempt to delete the original files
    #for file_ in files:
    #    try:
    #        os.remove(file_)
    #    except:
    #        # Ignore if the file cannot be deleted
    #        pass

    # Write the new FITS file to disk
    hdulist.writeto(path + output_label + '.fits',overwrite=True)
    # Attempt to delete the original files
    for file_ in files:
        try:
            os.remove(file_)
        except:
            # Ignore if the file cannot be deleted
            pass
    
        
def render_model_on_stamp(gal_pars,psf_pars, wcs, shape, g1=None, 
                          g2=None, nbrxyref=None, galaxy_model = 'bdf',psf_model = 'piff'):
    """
    Render an image of a galaxy using galaxy and PSF parameters, with optional shearing.

    Parameters:
        
        gal_pars (array-like): Parameters of the galaxy model.
        psf_pars (dict): Parameters of the PSF model. Can be a dictionary with 'turb' key for turbulent model.
        wcs (object): World Coordinate System object that contains transformation details.
        shape (int): The shape of the output image (assumed square).
        g1 (float, optional): First component of shear to apply to the galaxy image.
        g2 (float, optional): Second component of shear to apply to the galaxy image.
        return_PSF (bool, optional): Whether to return the PSF image along with the galaxy image.
        nbrxyref (tuple, optional): The reference (x, y) position of a neighbor to set the galaxy center.

    Returns:
        tuple: Depending on the return_PSF, it may return a galaxy image, PSF image, and a Jacobian object,
               or just the galaxy image and Jacobian object.

    Raises:
        Exception: If the image generation process fails.
    """
    

    if psf_model =='piff':
        psf_gmix = gmix.GMix(pars=psf_pars)
    elif psf_model =='turb':
        psf_gmix = ngmix.GMixModel(pars=psf_pars['pars'], model="turb")

    # Calculate the determinant of the WCS transformation matrix
    det = np.abs(wcs.getdet()) 

    # Initialize the Jacobian of the WCS transformation
    jac = ngmix.jacobian.Jacobian(row=wcs.xy0[1],
                                  col=wcs.xy0[0],
                                  dudrow=wcs.jac[0,1],
                                  dudcol=wcs.jac[0,0],
                                  dvdrow=wcs.jac[1,1],
                                  dvdcol=wcs.jac[1,0])

    if galaxy_model =='bdf':
        gmix_sky = gmix.GMixModel(gal_pars, model='bdf')
    elif galaxy_model =='shredder':
        g_ = copy.deepcopy(gal_pars)
        gmix_sky = ngmix.GMix(pars=g_.reshape(60))
        
    
    # Apply shear to the galaxy model if provided
    if (g1 is not None) and (g2 is not None):
        gmix_sky = gmix_sky.get_sheared(g1, g2)

    # Convolve the galaxy model with the PSF model
    gmix_image = gmix_sky.convolve(psf_gmix)

    # Attempt to generate the galaxy and PSF images
    try:
        # Generate the PSF image if requested
        #im_psf = psf_gmix.make_image((shape, shape), jacobian=jac) if return_PSF else None

        # Adjust the galaxy center if a neighbor reference position is provided
        if nbrxyref is not None:
            v, u = jac(nbrxyref[1], nbrxyref[0])
            gmix_image.set_cen(v, u)

        # Generate the galaxy image
        image = det*gmix_image.make_image((shape, shape), jacobian=jac)

    except Exception as e:
        # Handle exceptions during image generation and decide what to return based on return_PSF flag
        return None, jac

    # Return the requested images and Jacobian object
    return image, jac
    
    
def check_on_exposures(meds_array, exposure_list,bands):
    # Extract image paths from the meds_array, which is assumed to contain
    # metadata about the images. The '_image_info' dictionary key is used
    # to access the paths, and the leading character of each path is removed.
    
    images_path = [m._image_info['image_path'][1:] for m in meds_array]
    exposures_MEDS = dict()
    # Loop through each band (spectral range) to process the exposures.
    for index_band,band in enumerate(bands):
        # For the first band, extract the exposure numbers from the image paths.
        # The exposure number is assumed to be at the start of the file name,
        # following a specific pattern ('red/D00'). This number is converted to an integer.
        exposures_MEDS[index_band] = np.unique(np.array([np.int((image_path).split('_')[0].strip('red/D00')) for image_path in images_path[index_band]]))

    mask_exposure = dict()
    for index_band,band in enumerate(bands):
        # Determine which exposures in the current band are bad by matching them
        # against the exposure list obtained from the images.

        # these are lists of good exposures + CCD where the PSF did not fail! 
        match_exposures = np.array(np.in1d(exposure_list[band]['exp'], exposures_MEDS[index_band]))
        mask_exposure[band] = 100*exposure_list[band]['exp'][match_exposures] + exposure_list[band]['ccd'][match_exposures]

    # this is a list of EXP where the PSF FWHM is > 1.72 and that needs to be excluded.
    match_exposures = np.array(np.in1d(exposure_list['all']['exp'], exposures_MEDS[index_band]))
    mask_exposure['all'] = exposure_list['all']['exp'][match_exposures]
    return mask_exposure


def real_space_smoothing(x0, y0, shape, width=1.):
    """
    Create a real space smoothing kernel.

    This function generates a smoothing kernel based on cosine function, 
    typically used in image processing or data smoothing. The kernel 
    gradually transitions from a full weight to zero weight beyond 
    a specified radius.

    Parameters:
    x0, y0 (float): Coordinates of the center point for the kernel.
    shape (int): The size of the output square array (kernel).
    width (float): Scaling factor for the radius, default is 1.

    Returns:
    numpy.ndarray: A 2D array representing the smoothing kernel.
    """

    # Creating 1D arrays for x and y coordinates centered around 0
    x = np.arange(-shape//2, shape//2)
    y = np.arange(-shape//2, shape//2)

    # Creating 2D grids for x and y coordinates, shifted to center at (x0, y0)
    xx, yy = np.meshgrid(x - (x0 - shape//2), y - (y0 - shape//2))

    # Calculating the radius from the center (x0, y0) for each point in the grid
    rr = width * np.sqrt(xx**2 + yy**2)

    # Applying a smoothing function based on the radius
    # For radius < 10, the weight is 1 (full weight)
    # For radius between 10 and 20, the weight transitions using a cosine function
    # For radius > 20, the weight is 0 (no weight)
    real_weight = np.where(rr < 10, 1, 0.5 * (1 - np.cos(2 * np.pi * rr / (0.5 * 2 * 20))))
    real_weight = np.where(rr > 20, 0, real_weight)

    return real_weight



def save_moments_targets(self,fitsname):
    '''
    modified save function for moments with different sigma_Mf entries
    '''

    if len(self.id)>0:
        col=[]
        col.append(fits.Column(name="id",format="K",array=np.array(self.id)))


        col.append(fits.Column(name="ra",format="D",array= np.array(self.xy)[:,0]))
        col.append(fits.Column(name="dec",format="D",array=np.array(self.xy)[:,1]))


        col.append(fits.Column(name="moments",format="5E",array=np.array(self.moment)))  
        col.append(fits.Column(name="covariance",format="15E",array=np.array(self.cov).astype(np.float32)))
        col.append(fits.Column(name="covariance_psf_obs",format="15E",array=np.array(self.cov_psf_obs).astype(np.float32)))
        
            
        
        
        col.append(fits.Column(name="psf_moments",format="4E",array=self.psf_moment))
        col.append(fits.Column(name="psf_moment_obs",format="4E",array=self.psf_moment_obs))
        col.append(fits.Column(name="psf_hsm_moments",format="3E",array=self.psf_hsm_moment))
        col.append(fits.Column(name="psf_hsm_moments_obs",format="3E",array=self.psf_hsm_moments_obs))

        


        try:
            l = np.array(self.meb).shape[1]
        except:
            l = len(np.array(self.meb))
        col.append(fits.Column(name="bad_exposures",format="{0}K".format(l),array=np.array(self.bad_exposures)))
        col.append(fits.Column(name="good_exposures",format="{0}K".format(l),array=np.array(self.good_exposures)))

        col.append(fits.Column(name="Mf_per_band",format="{0}E".format(l),array=self.meb))
        col.append(fits.Column(name="cov_Mf_per_band",format="{0}E".format(l),array=np.array(self.cov_Mf_per_band)))
        col.append(fits.Column(name="mfrac_per_band",format="{0}E".format(l),array=np.array(self.mfrac_per_band)))

        col.append(fits.Column(name="DESDM_coadd_x",format="E",array=np.array(self.DESDM_coadd_x)))
        col.append(fits.Column(name="DESDM_coadd_y",format="E",array=np.array(self.DESDM_coadd_y)))


        try:
            l = np.array(self.orig_row).shape[1]
        except:
            l = len(self.orig_row)
        col.append(fits.Column(name="orig_col",format="{0}E".format(l),array=np.array(self.orig_col)))
        col.append(fits.Column(name="orig_row",format="{0}E".format(l),array=np.array(self.orig_row)))
        col.append(fits.Column(name="ccd_number",format="{0}K".format(l),array=np.array(self.ccd_name)))

        col.append(fits.Column(name="bkg",format="D",array=self.bkg))
        col.append(fits.Column(name="pixels_used_for_bkg",format="K",array=self.pixel_used_bkg))
        #'''
        # let's add, maybe, some average masked fraction

        self.prihdu.header['STAMPS'] = 0


        cols=fits.ColDefs(col)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)



     
class GalaxyModelsTable:
    """
    A class for handling galaxy models tables stored in FITS format.

    This class provides methods to load galaxy images and associated data
    from FITS files, allowing manipulation of the data such as computing
    moments, and accessing parameters for specific galaxies.

    Attributes:
        catalog (HDUList): An Astropy HDUList object representing the opened FITS file.
        this_is_wide_field (bool): Flag indicating whether the dataset is from a wide field (default True).
        cols (list): List of column names in the FITS file's second HDU.
        id_epoch_array (ndarray): Array of IDs for each epoch in the FITS file's third HDU.
        id_array (ndarray): Array of unique galaxy IDs in the FITS file's second HDU.
        bdf_mag (ndarray): Array of magnitudes for the galaxy models.
        bdf_flux (ndarray): Array of fluxes for the galaxy models.
        bdf_params (ndarray): Array of Bulge-Disk-Fit parameters for the galaxy models.
        pfs_params (ndarray): Array of point spread function parameters.
        bdf_ra (ndarray): Array of right ascension values for the galaxy models.
        bdf_dec (ndarray): Array of declination values for the galaxy models.
        numbands (int): Number of bands in the magnitude array.
        ra (ndarray): Array of right ascension values from the first HDU.
        dec (ndarray): Array of declination values from the first HDU.
    
    Methods:
        match_epochs_by_ID(ID): Returns the indices where the given ID matches the id_epoch_array.
        return_band_val_wide_field(bandind, indtostr): Converts band index to string or vice versa for wide fields.
        return_band_val_deep_fields(bandind, indtostr): Converts band index to string or vice versa for deep fields.
        return_model(band, pos, pos_epoch, shredder): Returns model parameters for a given galaxy and band.
        select_obj_by_ID(ID): Returns the indices where the given ID matches the id_array.
    """

    def __init__(self, path):
        """
        Initializes the GalaxyModelsTable object, loading data from a FITS file.

        Parameters:
            path (str): The file path to the FITS file containing galaxy model data.
            this_is_wide_field (bool, optional): Indicates if the data is from a wide field. Defaults to True.
        """

        self.catalog = fits.open(path)
        self.cols = self.catalog[1].data.columns.names
        self.id_epoch_array = self.catalog[2].data['id']
        self.id_array = self.catalog[1].data['id']
        self.bdf_mag = self.catalog[1].data['bdf_mag']
        self.bdf_flux = self.catalog[1].data['bdf_flux']
        self.bdf_params = self.catalog[1].data['bdf_pars']
        self.pfs_params = self.catalog[2].data['psf_pars']
        self.bdf_ra = self.catalog[1].data['ra']
        self.bdf_dec = self.catalog[1].data['dec']
        self.numbands = np.shape(self.bdf_mag)[1]
        self.ra = self.catalog[1].data['ra']
        self.dec = self.catalog[1].data['dec']
        

    def match_epochs_by_ID(self, ID):
        """Returns the indices of epochs with the specified ID."""
        return np.where(self.id_epoch_array == ID)


    
    def return_band_val(self,bandind,indtostr=True):
        if indtostr:
            # this is for wide field
            if self.numbands==5:
                if bandind==0: return 'g'
                if bandind==1: return 'r'
                if bandind==2: return 'i'
                if bandind==3: return 'z'
                if bandind==4: return 'Y'
           # this is for deep field
            elif self.numbands==4:
                if bandind==0: return 'g'
                if bandind==1: return 'r'
                if bandind==2: return 'i'
                if bandind==3: return 'z'
        else:
            if self.numbands==5:

                if bandind=='g': return 0
                if bandind=='r': return 1
                if bandind=='i': return 2
                if bandind=='z': return 3
                if bandind=='Y': return 4
            elif self.numbands==4:
                if bandind=='g': return 0
                if bandind=='r': return 1
                if bandind=='i': return 2
                if bandind=='z': return 3

    def return_model(self, band='i', pos=None, pos_epoch=None, shredder=False):
        """
        Returns model parameters for a specified galaxy and band.

        Parameters:
            band (str): The band for which model parameters are requested.
            pos (int): The position index in the bdf_params and bdf_flux arrays.
            pos_epoch (int): The position index in the pfs_params array.
            shredder (bool): Flag indicating whether to use the shredder algorithm (not yet implemented).

        Returns:
            dict: A dictionary containing 'gal_pars' and 'psf_pars' for the galaxy.
        """
        index_band = self.return_band_val(band, indtostr=False)
        #else:
        #    index_band = self.return_band_val_deep_fields(band, indtostr=False)

        gal_pars = self.bdf_params[pos][0:6]
        gal_pars = np.append(gal_pars, self.bdf_flux[pos][index_band])
        pos_epoch = pos_epoch[index_band]
        psf_pars = self.pfs_params[pos_epoch]
        mof_params = {'gal_pars': gal_pars, 'psf_pars': psf_pars}

        return mof_params
            
    def select_obj_by_ID(self, ID):
        """Returns the indices of objects with the specified ID."""
        return np.where(self.id_array == ID)

    
    


class CollectionOfImages:
    '''
    This class is designed to manage and process a collection of images from the Multi-Epoch Data Structure (MEDS). It supports loading images into memory, manipulating them, and performing operations like measuring moments and rendering models.
    '''

    def __init__(self):
        '''
        Initializes the CollectionOfImages instance.
        - MEDS_stamps: List to store MEDS image data.
        - coadd_IDs: List to store the coaddition IDs associated with each image.
        - MEDS_indexes: List to store indexes of MEDS images.
        '''
        self.MEDS_stamps = []
        self.coadd_IDs = []
        self.MEDS_indexes = []
        
        
    def add_MEDS_stamp(self, MEDS_stamp):
        '''
        Adds a single MEDS stamp to the collection.
        - MEDS_stamp: The MEDS image data to be added.
        Each MEDS stamp's coadd ID and MEDS index are also stored.
        '''
        self.MEDS_stamps.append(MEDS_stamp)
        self.coadd_IDs.append(MEDS_stamp.coadd_ID[0])
        self.MEDS_indexes.append(MEDS_stamp.MEDS_index)

        
       
        
        
    def add_models(self, galaxy_models_table):
        '''
        Reads model parameters from a GalaxyModelsTable and stores them into the corresponding MEDS_stamp objects.
        - galaxy_models_table: The table containing galaxy model parameters.
        This function matches the models to the MEDS stamps based on their IDs, and handles multiple entries for epochs.
        '''
        
        # generate columns of matched positions
        index_to_match = np.arange(len(galaxy_models_table.id_array))
        data_ = {'pos': index_to_match}
        df1 = pd.DataFrame(data = data_, index = galaxy_models_table.id_array)
        df2 = pd.DataFrame(index = self.coadd_IDs)   
        
        self.pos = np.array(df2.join(df1).loc[self.coadd_IDs,'pos'])
        self.model_index = np.array(df2.join(df1).loc[self.coadd_IDs].index)
   
        # generate columns of matched positions for the epochs, which are in another table
        # and have multiple entries
        index_to_match = np.arange(len(galaxy_models_table.id_epoch_array))
        df1 = pd.DataFrame(data = {'pos': index_to_match} , index = galaxy_models_table.id_epoch_array)
        self.pos_epoch = (df2.join(df1))


                                  
                                  
        for MEDS_stamp, pos, model_index in zip(self.MEDS_stamps, self.pos, self.model_index):
            
            models_parameters = dict()
            
            try:
                pos_epoch = (np.array(self.pos_epoch.loc[ MEDS_stamp.coadd_ID[0]]).astype(np.int))[:,0]
            except:
                pos_epoch = (np.array(self.pos_epoch.loc[ MEDS_stamp.coadd_ID[0]]).astype(np.int))[0]
                
            for band in MEDS_stamp.bands:
                try:
                    models_parameters[band] = galaxy_models_table.return_model(band = band, pos = pos, pos_epoch = pos_epoch)
                except:
                    print ('failed rendering model for entry {0}, band [{1}]'.format( MEDS_stamp.MEDS_index,band))
                    MEDS_stamp.flags += 2
    
                
            MEDS_stamp.model_parameters = models_parameters 
            
            



        
    def render_models(self, MEDS_index = 0, render_self = False, render_others = True, use_COADD_only = True):
        '''
        Renders models onto the MEDS stamps.
        - MEDS_index: Index of the MEDS stamp to render models on.
        - render_self: Flag to indicate if the model of the MEDS stamp itself should be rendered.
        - render_others: Flag to indicate if models of other objects should be rendered.
        - use_COADD_only: Flag to use only the coaddition image for rendering.
        This function handles the rendering of models onto the image stamps, including both the target object and others in the field.
        '''

        model_rendered = []
        model_rendered_all = []
        
        flag_rendered_models = []
        
        for b, band in enumerate(self.MEDS_stamps[MEDS_index].bands):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.MEDS_stamps[MEDS_index].ncutout[b]

            model_rendered_band = []
            model_rendered_all_band = []
            
            
            list_MEDS_indexes = np.unique(self.MEDS_stamps[MEDS_index].seglist[b][0].flatten())
            list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=0]
            list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=self.MEDS_stamps[MEDS_index].MEDS_index+1]

            flag_rendered_models_band = []
            
            # sometimes the model doesn't exist. in this case, let's flag the exposure 
            for i in range( end): 
                if i >= start:
                    try:
                        size_image = self.MEDS_stamps[MEDS_index].imlist[b][i].shape[0]    

                        rendered_image = np.zeros((size_image,size_image))


                        # generate self model -----
                        wcs = self.MEDS_stamps[MEDS_index].wcslist[b][i]
                        psf_gmix = ngmix.gmix.GMix(pars=self.MEDS_stamps[MEDS_index].model_parameters[band]['psf_pars'])
                        det=np.abs(wcs.getdet()) 
                        #'''
                        if (self.MEDS_stamps[MEDS_index].mfrac_inner[b][i] > 0) and (self.MEDS_stamps[MEDS_index].mfrac_flag[b][i]) :


                            gmix_sky  = ngmix.gmix.GMixBDF(self.MEDS_stamps[MEDS_index].model_parameters[band]['gal_pars'])
                            det = np.abs(wcs.getdet()) 
                            jac = ngmix.jacobian.Jacobian(  row=wcs.xy0[1],
                                                            col=wcs.xy0[0],
                                                            dudrow=wcs.jac[0,1],
                                                            dudcol=wcs.jac[0,0],
                                                            dvdrow=wcs.jac[1,1],
                                                            dvdcol=wcs.jac[1,0])
                            gmix_image = gmix_sky.convolve(psf_gmix)


                            #v, u = jac(nbrxyref[1], nbrxyref[0])
                            #gmix_image.set_cen(v, u)
                            image_self = det*gmix_image.make_image((size_image, size_image), jacobian=jac)

                        else:

                            image_self = None
                        #'''
                        image_self = None
                    
                        central_went_wrong = False
                    except:
                        image_self = None
                        flag_rendered_models_band.append(False)
                        central_went_wrong = True
                    # identify neighbours

                    flag_rendered_models_band_i = True
                    if not central_went_wrong:
                        for mute_index in list_MEDS_indexes:
                            try:
                                MEDS_index_j = mute_index-1

                                st = timeit.default_timer()
                                
                                pos_band = np.arange(self.MEDS_stamps[MEDS_index].n_bands)[np.in1d(self.MEDS_stamps[MEDS_index].bands,band)][0]

                                drow = self.MEDS_stamps[MEDS_index].rowcol[pos_band][i][0] - (self.MEDS_stamps[MEDS_index].orig_rowcol[pos_band][i][0] - self.MEDS_stamps[MEDS_index_j].orig_rowcol[pos_band][i][0])
                                dcol = self.MEDS_stamps[MEDS_index].rowcol[pos_band][i][1] - (self.MEDS_stamps[MEDS_index].orig_rowcol[pos_band][i][1] - self.MEDS_stamps[MEDS_index_j].orig_rowcol[pos_band][i][1])


                                jac = ngmix.jacobian.Jacobian(row=drow,
                                                        col=dcol,
                                                        dudrow=wcs.jac[0,1],
                                                        dudcol=wcs.jac[0,0],
                                                        dvdrow=wcs.jac[1,1],
                                                        dvdcol=wcs.jac[1,0])

                                pars_ = self.MEDS_stamps[MEDS_index_j].model_parameters[band]['gal_pars']
                                if pars_[-1] <0 :
                                    pars_[-1] = 0.
                                gmix_sky  = ngmix.gmix.GMixBDF(pars_)

                                gmix_image = gmix_sky.convolve(psf_gmix)
                                image = det*gmix_image.make_image((size_image, size_image), jacobian=jac)

                                rendered_image+=image
                                flag_rendered_models_band_i = flag_rendered_models_band_i & True
                                
                                end = timeit.default_timer()

                            except:
                                flag_rendered_models_band_i = flag_rendered_models_band_i & False
                                
                                pass
                        flag_rendered_models_band.append(flag_rendered_models_band_i)
                
                    model_rendered_band.append(rendered_image)
                    if image_self is not None:
                        rendered_image += image_self
                    model_rendered_all_band.append(rendered_image)
                else:
                    flag_rendered_models_band.append(False)
                    model_rendered_band.append(None) 
                    model_rendered_all_band.append(None)    
                    
                

            model_rendered.append(model_rendered_band) 
            model_rendered_all.append(model_rendered_all_band) 
            flag_rendered_models.append(flag_rendered_models_band)
                           
        self.MEDS_stamps[MEDS_index].model_rendered = model_rendered
        self.MEDS_stamps[MEDS_index].model_all_rendered = model_rendered_all
        self.MEDS_stamps[MEDS_index].flag_rendered_models =   flag_rendered_models



        
        # check that there are enough exposures
        
    
        bands_not_masked = dict()
        for b in range(len(self.MEDS_stamps[MEDS_index].bands)):
            bands_not_masked[self.MEDS_stamps[MEDS_index].bands[b]] = True
            if use_COADD_only:
                if not flag_rendered_models[b][i]:
                    bands_not_masked[self.MEDS_stamps[MEDS_index].bands[b]] = False
                    
            else:
                bands_not_masked[self.MEDS_stamps[MEDS_index].bands[b]] = False
                for i in range(1, self.MEDS_stamps[MEDS_index].ncutout[b]):  #MG *******+
                    # at least one exposure needs to make it
                    if not flag_rendered_models[b][i]:
                        pass
                    else:
                        bands_not_masked[self.MEDS_stamps[MEDS_index].bands[b]] = True
                  
        bands_not_masked_list = []
        for b in bands_not_masked.keys():
            if bands_not_masked[b]:
                 bands_not_masked_list.append(b)
        return bands_not_masked_list

        

class MedsStamp:
    '''
    This class is designed to manage and process individual MEDS stamps, which are subsections of astronomical images. It facilitates the computation of various quantities relevant to the BFD (Balanced Fourier Descriptor) pipeline, such as moments, PSD (Power Spectral Density) parameters, etc.
    '''

    def __init__(self, MEDS_index, meds_array=[], bands=[]):
        '''
        Initializes the MedsStamp instance.
        - MEDS_index: Index of the MEDS stamp in the larger MEDS array.
        - meds_array: Array containing MEDS data.
        - bands: List of bands (spectral ranges) associated with the MEDS stamp.
        Attributes like coadd_ID, xyshift, moments, flags, and model parameters are also initialized.
        '''
        self.MEDS_index = MEDS_index
        self.coadd_ID = [m['id'][MEDS_index] for m in meds_array]
        
        self.bands = bands
        
        # this is useful to know in advance for the neighbours subtraction
        self.ncutout = [m['ncutout'][MEDS_index] for m in meds_array]
        self.orig_rowcol = [[ (meds_array[b]['orig_row'][MEDS_index][i],meds_array[b]['orig_col'][MEDS_index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
            
            
        self.n_bands = len(bands)
        self.xyshift = None
        self.moments = None
        self.flags = 0
        self.model_parameters = None
        self.index_model = None

    def check_stamp_masked_frac(self, limit=0.1, use_COADD_only = False):
        '''
        Checks which bands have exposures with a fraction of pixels masked above a specified limit.
        - limit: The threshold for considering a pixel as masked.
        - use_COADD_only: Flag to consider only the coadded image.
        Returns a list of bands not significantly masked.
        '''
        
  
        self.size =  [[len(self.masklist[b][i].flatten()) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        #self.mfrac = [[1.*len(np.zeros(self.size[b][i])[(self.wtlist[b][i].flatten()==0.) | (self.masklist[b][i].flatten()!=0.)])/self.size[b][i] for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        self.mfrac_flag = [[True for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        #ed = timeit.default_timer()
        #
        
        # check the size of the stamps
        sizes = []
        for b in range(len(self.bands)):
            for i in range((self.ncutout[b])):
                sizes.append(self.wtlist[b][i].shape[0])
        min_size = min(np.array(sizes))
        
        # size of Gaussian
        if min_size>40:
            size = 40
        elif min_size>30:
            size = 30
        elif min_size>20:
            size = 20
        Gaussian_weight = galsim.Gaussian(fwhm=2).drawImage(nx=size,ny=size, scale=0.263)
        Gaussian_weight.setCenter(0,0)

        mfrac = []
        mfrac_inner = []
        for b in range(len(self.bands)):
            mfrac_ = []
            mfrac_inner_ = []
            for i in range((self.ncutout[b])):

                self.wtlist[b][i]
                
                shape = self.imlist[b][i].shape

                # add max()
                slice1 = slice((shape[0]-size)//2, shape[0]- (shape[0]-size)//2)
                slice2 = slice((shape[0]-size)//2, shape[0]- (shape[0]-size)//2)
                mask_ = ((self.wtlist[b][i][slice1,slice2]==0.) | (self.masklist[b][i][slice1,slice2] != 0.))
                mfrac_.append(np.sum((Gaussian_weight.array)[mask_])/np.sum((Gaussian_weight.array)))
                
                
                mask_ = mask_ & (Gaussian_weight.array>0.2*np.max(Gaussian_weight.array))
                mfrac_inner_.append(np.sum((Gaussian_weight.array)[mask_])/np.sum((Gaussian_weight.array)))
                
                
                
            mfrac.append(mfrac_)
            mfrac_inner.append(mfrac_inner_)
        self.mfrac = mfrac
        self.mfrac_inner = mfrac_inner
                
                
  
        #mfrac_per_band = []
        #for b in range(len(self.bands)):
        #    mfrac_per_band.append(np.mean(np.array(self.mfrac[b])))
        #self.mfrac_per_band = mfrac_per_band
        bands_not_masked = dict()
        mfrac_per_band = np.zeros(len(self.bands))
        
        mfrac_per_band_w = np.zeros(len(self.bands))
        for b in range(len(self.bands)):
            bands_not_masked[self.bands[b]] = True
            if use_COADD_only:
                if (self.mfrac[b][0] > limit):
                    bands_not_masked[self.bands[b]] = False
                    self.mfrac_flag[b][0] = False
                else:
                    mfrac_per_band[b] += self.mfrac[b][0]
                    mfrac_per_band_w[b] += 1
            else:
                bands_not_masked[self.bands[b]] = False
                for i in range(1, self.ncutout[b]):  #MG *******+
                    # at least one exposure needs to make it
                    if (self.mfrac[b][i] > limit):
                        self.mfrac_flag[b][i] = False
                    else:
                        bands_not_masked[self.bands[b]] = True
                        
                        mfrac_per_band[b] += self.mfrac[b][0]
                        mfrac_per_band_w[b] += 1
                  
        self.mfrac_per_band = (mfrac_per_band/mfrac_per_band_w)
        self.mfrac_per_band[mfrac_per_band_w==0] = -1.
        
        bands_not_masked_list = []
        for b in bands_not_masked.keys():
            if bands_not_masked[b]:
                 bands_not_masked_list.append(b)
        self.bands_not_masked = bands_not_masked_list
        return bands_not_masked_list



        
    def compute_moments(self, sigma = 3, FFT_pad_factor = 2., use_COADD_only = False, bands = ['g','r', 'i', 'z'], 
                                  bands_weights = [0.,0.7,0.2,0.1] , Detectinator_=False):
        '''
        Computes moments by combining exposures and bands.
        - sigma, FFT_pad_factor: Parameters for the BFD filter.
        - use_COADD_only: Flag to consider only the coadded image.
        - bands: List of bands to consider.
        - bands_weights: Weights for different bands.
        - Detectinator_: Flag to use a specialized detection algorithm.
        This method is key for extracting scientific information from the image.
        '''
        
        BFD_filter = mc.KBlackmanHarris(sigma = sigma) 
        
        images_array = []
        psf_array = []
        wcs_array = []
        noise_array = []
        band_array = []
        
        psf_moments = np.zeros(4)
        psf_weights = 0
        
        good_exposures = 0
        bad_exposures = 0
        for  index_band,band in enumerate(bands):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[index_band]
                
            for exp in range(start, end):  
                compute = True
                # check if we can use the exposure
                if self.mfrac_flag[index_band][exp]:
                    pass
                else:
                    compute = compute & False
                if self.explist_mask is not None:
                    if self.explist_mask[index_band][exp-1]:
                        pass
                    else:
                        compute = compute & False
            
                if self.flag_rendered_models[index_band][exp]:
                    pass
                else:
                    compute = compute & False
                    
                if compute:
                    good_exposures += 1
                    img = self.imlist[index_band][exp] - self.model_rendered[index_band][exp]
                    

                    images_array.append(img)
                    
                    psf_array.append( self.psf[index_band][exp])
                    
                    noise_rms = (1./np.sqrt(np.median(self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])))
                    noise_array.append(noise_rms)
                    
                    band_array.append(self.bands[index_band])
                    
                    
                    
                    # Detectinator ---------------------------------------------------------------------------------
                    '''     
                    I think this has to be 1 coherent shift for th galaxy exposures, and proabbly should be a weighted average of all the
                    exposures shifts, or something like that.
                    duv_dxy = np.array([[self.wcslist[index_band][exp].jac[0,0], self.wcslist[index_band][exp].jac[0,1]],
                                        [self.wcslist[index_band][exp].jac[1,0], self.wcslist[index_band][exp].jac[1,1]]])
                    
                    
                    dx,dx_init,kval,ku,kv,d2k,conjugate,kvar,mf_map,mf_cov = Detectinator(img,
                                                                          psf=self.psf[index_band][exp],
                                                                          noise_rms=noise_rms,
                                                                          sn=3,
                                                                          duv_dxy=duv_dxy,
                                                                          minsep=5)
                    wcs_exposure = copy.deepcopy(self.wcslist[index_band][exp])
    
                    try:
                        shift_ =  (dx[0][0]-dx_init[0][0])**2+(dx[0][1]-dx_init[0][1])**2
                        wcs_exposure.xy0 = dx[0]
                        wcs_array.append(wcs_exposure)
                    except:
                        wcs_array.append(self.wcslist[index_band][exp])
                    '''
                    wcs_array.append(self.wcslist[index_band][exp])
                else:
                    bad_exposures +=1
              

        #self.good_exposures = good_exposures      
        #self.bad_exposures = bad_exposures  
        
        # compute image moments ----
        kds, psf_shifts = bfd.multiImage(images_array, (0,0), psf_array, wcs_array, 
                             pixel_noiselist = noise_array, bandlist = band_array,
                             pad_factor=FFT_pad_factor, psf_recenter_sigma = 2.)
        
        
        # Compute PSF moments, for diagnosticss
             
        wcs_array_corrected = []
        delta_stamp_array = []
        psf_array_2 = []
        for i in range(len(kds)):
            nominal = np.array(psf_array[i].shape) // 2
            origin = (0.,0.)

                        

            wcs_psf = wcs_array[i]
            wcs_psf.xy0 = (nominal+[psf_shifts[i][1],psf_shifts[i][0]])
            wcs_array_corrected.append(wcs_psf)

            delta_stamp = np.zeros_like(psf_array[i])
            delta_stamp[nominal[0],nominal[1]] = 1.
            delta_stamp_array.append(delta_stamp)
            

        psf_kd, _  = bfd.multiImage(psf_array, (0,0), delta_stamp_array, wcs_array_corrected, 
                     pixel_noiselist = noise_array, bandlist = band_array,
                     pad_factor=FFT_pad_factor) 
            

            
        bandinfo = {'bands':bands, 'weights':bands_weights,'index': np.arange(len(bands))} 
        psf_moment = bfd.MultiMomentCalculator(psf_kd, BFD_filter, bandinfo = bandinfo)
        psf_moment.recenter()
        Mf,Mr,M1,M2,_ = psf_moment.get_moment(0.,0.).even
        self.psf_moments  = np.array([Mf,Mr,M1,M2])
        

        multi_moment = bfd.MultiMomentCalculator(kds, BFD_filter, bandinfo = bandinfo)

        self.xyshift, error,msg = multi_moment.recenter()
        self.moments = multi_moment
        
 
        
    def compute_moments_observed_psf(self, sigma = 3, FFT_pad_factor = 2., use_COADD_only = False, bands = ['g','r', 'i', 'z'], 
                                  bands_weights = [0.,0.7,0.2,0.1] , Detectinator_=False):
        '''
        Computes moments by combining exposures and bands. Special case where we observed a star.
        - sigma, FFT_pad_factor: Parameters for the BFD filter.
        - use_COADD_only: Flag to consider only the coadded image.
        - bands: List of bands to consider.
        - bands_weights: Weights for different bands.
        - Detectinator_: Flag to use a specialized detection algorithm.
        This method is key for extracting scientific information from the image.
        '''
        
        BFD_filter = mc.KBlackmanHarris(sigma = sigma) 
        
        images_array = []
        psf_array = []
        wcs_array = []
        noise_array = []
        band_array = []
        delta_stamp_array = []
        
        psf_moments = np.zeros(4)
        psf_weights = 0
        
        for  index_band,band in enumerate(bands):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[index_band]
                
            for exp in range(start, end):  
                compute = True
                # check if we can use the exposure
                if self.mfrac_flag[index_band][exp]:
                    pass
                else:
                    compute = compute & False
                if self.explist_mask is not None:
                    if self.explist_mask[index_band][exp-1]:
                        pass
                    else:
                        compute = compute & False
            
                if self.flag_rendered_models[index_band][exp]:
                    pass
                else:
                    compute = compute & False
                    
                if compute:
                    
                    img = self.imlist[index_band][exp] - self.model_rendered[index_band][exp]
                    #xy0 = self.wcslist[index_band][exp].xy0
                    #smoothing = real_space_smoothing(xy0[0], xy0[1], img.shape[0], width=2)
                    
                    images_array.append(img) #*smoothing)
                    
                    noise_rms = (1./np.sqrt(np.median(self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])))
                    noise_array.append(noise_rms)
                    
                    band_array.append(self.bands[index_band])
                    
                    wcs_array.append(self.wcslist[index_band][exp])
                        
                    nominal = np.array(self.imlist[index_band][exp].shape) // 2
                    delta_stamp = np.zeros_like(self.imlist[index_band][exp])
                    delta_stamp[nominal[0],nominal[1]] = 1.
                    delta_stamp_array.append(delta_stamp)


        psf_kd, _  = bfd.multiImage(images_array, (0,0), delta_stamp_array, wcs_array, 
                     pixel_noiselist = noise_array, bandlist = band_array,
                     pad_factor=FFT_pad_factor) 
            
            
            
        bandinfo = {'bands':bands, 'weights':bands_weights,'index': np.arange(len(bands))} 
        psf_moment = bfd.MultiMomentCalculator(psf_kd, BFD_filter, bandinfo = bandinfo)
        psf_moment.recenter()
        Mf,Mr,M1,M2,_ = psf_moment.get_moment(0.,0.).even
        self.psf_moments_observed  = np.array([Mf,Mr,M1,M2])

        covm_even,covm_odd , covm_even_all , _ = psf_moment.get_covariance(returnbands=True)
        covgal = MomentCovariance(covm_even,covm_odd)
        
 
                
        self.psf_moments_observed_cov  = covgal.pack()

        
    def compute_noise(self):
        '''
        Computes the noise level in the image based on weight maps and masks.
        '''
        self.noise_rms =  [[np.sqrt(1./np.median(self.wtlist[b][i][(self.wtlist[b][i]> 0.) & (self.masklist[b][i] == 0)])) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
 
            
    
    def deal_with_bmask(self, use_COADD_only = False):
        '''
        Processes the bitmask of the image to handle masked or bad pixels.
        - use_COADD_only: Flag to consider only the coadded image.

        #if mfrac_inner>0, let's use the rendered model, otherwise just add random noise to the image.
        '''
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


                self.imlist[b][i][bmask!=0] = self.model_all_rendered[b][i][bmask!=0]
                s0 = self.model_all_rendered[b][i].shape[0]
                rnd_ = np.random.normal(size = (s0,s0))

                self.imlist[b][i][(bmask!=0) & (self.wtlist[b][i]>0.)] += rnd_[(bmask!=0) & (self.wtlist[b][i]>0.)]/np.sqrt(self.wtlist[b][i])[(bmask!=0) & (self.wtlist[b][i]>0.)]

                ave_noise = np.sqrt(np.median(self.wtlist[b][i][self.wtlist[b][i]>0.]))
                self.imlist[b][i][self.wtlist[b][i]<=0.] += rnd_[(self.wtlist[b][i]<=0.)]/ave_noise




        
        
    def Load_MEDS(self, index, meds_array = [], mask_exposure = None, psf_array = None):
        '''
        Loads MEDS data into memory for a specific detection.
        It processes images, segmentation maps, weight maps, bitmasks, Jacobians, and PSFs for different exposures.
        - index: Index of the MEDS stamp.
        - meds_array: Array containing MEDS data.
        - mask_exposure: Optional mask to filter out bad exposures.
        n.b: First cutout for each band is always the COADD
        '''
        
        if meds_array != None:
            self.image_ra = [m['ra'][index] for m in meds_array]
            self.image_dec = [m['dec'][index] for m in meds_array]
            #self.ncutout = [m['ncutout'][index] for m in meds_array]
            self.imlist = [m.get_cutout_list(index) for m in meds_array]
   

            #self.orig_rowcol = [[ (meds_array[b]['orig_row'][index][i],meds_array[b]['orig_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
            self.orig_start_rowcol = [[ (meds_array[b]['orig_start_row'][index][i],meds_array[b]['orig_start_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
     
            # row,col in the cutouts. Is generally close to center of the tile.
            self.rowcol = [[meds_array[b].get_cutout_rowcol(index,i) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
            self.seglist = [m.get_cutout_list(index, type='seg') for m in meds_array]
            self.wtlist = [m.get_cutout_list(index, type='weight') for m in meds_array]
            self.masklist = [m.get_cutout_list(index, type='bmask') for m in meds_array]
            self.jaclist = [m.get_jacobian_list(index) for m in meds_array]
            

            #get ccd numbers ----
            self.ccd_name =  [[ 1000*b+np.int((meds_array[b]._image_info['image_path'][meds_array[b]['file_id'][index][i]]).split('_')[2].strip('c')) for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))]  
            self.expnum =  [[ np.int((meds_array[b]._image_info['image_path'][meds_array[b]['file_id'][index][i]]).split('_')[0].strip('red/D00')) for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))] 
          

            # Make a mask in case any of the exposures are in the bad exposures list
           # '''
            if mask_exposure is not None:        
                self.expnum_ccd =  [[ (self.ccd_name[b][i-1]-1000*b)+100*self.expnum[b][i-1]  for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))] 

                
                # the first entry is a check on good exposures, the second on bad exposures.
                self.explist_mask =  [[ ((self.expnum_ccd[b][i-1] in mask_exposure[self.bands[b]]) and (not(self.expnum[b][i-1] in mask_exposure['all'])))  for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))]

            else:
                self.explist_mask = None
            

            # save DESDM coordinates ----
            self.DESDM_coadd_x = [m['input_row'][index] for m in meds_array]
            self.DESDM_coadd_y = [m['input_col'][index] for m in meds_array]
            
            

            if psf_array is not None:
                # this is to speed up the deep fields
                self.psf = [[ psf_array[i][m._cat['psf_start_row'][index, icut]:m._cat['psf_start_row'][index, icut]+25*25].reshape(25,25)  for icut in range(m['ncutout'][index]) ] for i,m in enumerate(meds_array)]
               
            else:
                try:
                    self.psf = [m.get_cutout_list(index, type='psf') for m in meds_array]
                except:
                    if self.verbose > 0:
                        print ('No PSF extension in this file')

            #end = timeit.default_timer()
      

        
        
    def make_WCS(self):
        '''
        Generates WCS (World Coordinate System) information for the MEDS stamp.
        This is essential for astronomical image processing to understand the spatial orientation of the image.
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
            
            
            

  
            

   
        

            
    def measure_psf_HSM_moments(self, bands_weights = [], use_COADD_only = False, do_it_for_the_image=False):
        
        psf_moments = np.zeros(4)
        psf_moments_w = np.zeros(4)
        for b in range(len(self.bands)): 

            
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[b]
                
            for i in range(start,end):
                y = self.orig_rowcol[b][i][0]+1
                x = self.orig_rowcol[b][i][1]+1

                try:
                #if 1==1:
                    if do_it_for_the_image:
                        img = galsim.Image(self.imlist[b][i]-self.model_rendered[b][i],copy=True)
                    else:
                        img = galsim.Image(self.psf[b][i],copy=True)

                    jac = self.jaclist[b][i]
                    jac = galsim.JacobianWCS(dudx=jac['dudcol'], dudy=jac['dudrow'], dvdx=jac['dvdcol'], dvdy=jac['dvdrow'])


                    # calculate hsm moments -----
                    try:
                    #if 1==1:
                        mom = galsim.hsm.FindAdaptiveMom(img)
                        sigma = mom.moments_sigma
                        shape = mom.observed_shape
                        scale, shear, theta, flip = jac.getDecomposition()
                        sigma *= scale
                        # Fix shear.  First the flip, if any.
                        if flip:
                            shape = galsim.Shear(g1 = -shape.g1, g2 = shape.g2)
                        # Next the rotation
                        shape = galsim.Shear(g = shape.g, beta = shape.beta + theta)
                        # Finally the shear
                        shape = shear + shape
                        e1 = shape.g1
                        e2 =  shape.g2

                        ssq = sigma**2 / (1-e1**2-e2**2)**0.5
                        M = [[(1+e1)*ssq, e2*ssq],[e2*ssq, (1-e1)*ssq]]


                        weight = bands_weights[b]

                        psf_moments += np.array([weight*ssq,weight*M[0][0],weight*M[1][1],weight*M[0][1]])
                        psf_moments_w += weight*np.array([1,1,1,1])
                        #print (psf_moments,psf_moments_w)
                    except:
                        pass
                except:
                    pass
        psf_moments = psf_moments/psf_moments_w

        # compute psf ellipticities:    
        sigma_ = np.sqrt(np.sqrt(psf_moments[1]*psf_moments[2]-psf_moments[3]**2))
        e1_ = psf_moments[1]/psf_moments[0]-1
        e2_ = psf_moments[3]/psf_moments[0]
 
        if do_it_for_the_image:
            self.psf_hsm_moments_obs = [sigma_,e1_,e2_]
        else:
            self.psf_hsm_moments = [sigma_,e1_,e2_]

        
   
            
    def number_of_good_exposures_per_band(self, use_COADD_only=False):
        '''
        Calculates the number of good exposures for each band in the dataset.

        Parameters:
        use_COADD_only (bool): If True, the function only considers co-added exposures.
                               If False, it considers all exposures except the co-added ones.

        Returns:
        np.ndarray: An array containing the number of good exposures per band.

        The function goes through each band and evaluates certain criteria to determine
        if an exposure is 'good'. These criteria involve various flags indicating the 
        quality or usability of the exposure.
        '''

        # Initialize an array to count good exposures for each band
        bands_good_exposures = np.zeros(len(self.bands)).astype(int)
        bands_bad_exposures = np.zeros(len(self.bands)).astype(int)

        # Iterate over each band
        for index_band, band in enumerate(self.bands):
            # Determine the range of exposures to consider based on the use_COADD_only flag
            if use_COADD_only:
                start, end = 0, 1
            else:
                start, end = 1, self.ncutout[index_band]

            # Iterate over each exposure within the specified range
            for exp in range(start, end):
                compute = True  # Initialize a flag to determine if the exposure is good

                # Check various conditions to see if the exposure should be counted

                # If mfrac_flag for this exposure is False, skip the exposure
                if self.mfrac_flag[index_band][exp]:
                    pass
                else:
                    compute = compute & False

                # If explist_mask is not None and the mask for this exposure is False, skip it
                if self.explist_mask is not None:
                    if self.explist_mask[index_band][exp - 1]:
                        pass
                    else:
                        compute = compute & False

                # If flag_rendered_models for this exposure is False, skip the exposure
                if self.flag_rendered_models[index_band][exp]:
                    pass
                else:
                    compute = compute & False
                    
                    
                # If background_subtraction_OK_flag for this exposure is True, keep the exposure
                if self.background_subtraction_OK_flag[index_band][exp]:
                    pass
                else:
                    compute = compute & False
                    

                # If all conditions are met (compute is still True), count this as a good exposure
                if compute:
                    bands_good_exposures[index_band] += 1
                else:
                    bands_bad_exposures[index_band] += 1
                    

        # Return the count of good exposures for each band
        return bands_good_exposures   , bands_bad_exposures       
            
    def return_orig_coordinates(self):
        '''
        Returns the original coordinates of the image within its larger context (e.g., the full astronomical survey).
        Useful for mapping the stamp back to its source location.
        '''
        orig_row_flattened = np.zeros(70)
        orig_col_flattened = np.zeros(70)
        ccd_name_flattened = -np.ones(70)


        u_1 = np.array([ll[0] for l in self.orig_rowcol for ll in l[1:]])
        u_2 = np.array([ll[1] for l in self.orig_rowcol for ll in l[1:]])
        u_3 = np.array([ll for l in self.ccd_name for ll in l])


        orig_row_flattened[:len(u_1)] = u_1 
        orig_col_flattened[:len(u_1)] = u_2 
        ccd_name_flattened[:len(u_1)] = u_3 
        return orig_row_flattened,orig_col_flattened, ccd_name_flattened
    
            
            
            
    def subtract_background(self, use_COADD_only = False):
        '''
        Recomputes and subtracts the background from the image based on the outermost pixels.
        This method is used for background noise reduction.
        '''
        
         


        bkg_tot = 0
        count = 0
        len_v = 0
        
        self.background_subtraction_OK_flag = [[True for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        

        
        for b, band in enumerate(self.bands):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[b]

                
            for i in range(start,end):

            
                try:
                    mask_cond = (self.masklist[b][i] == 0 ) & (self.model_all_rendered[b][i]<0.01*self.noise_rms[b][i])
                except:
               
                    mask_cond = ( self.masklist[b][i] == 0 )
                    
                    
                #mask_cond = (self.model_all_rendered[b][i]>0.01*self.noise_rms[b][i])
                    
                # Update mask condition
                maskarr = np.ones_like(mask_cond, dtype='int')
                maskarr[~mask_cond] = 0
                uu = 6
                maskarr[uu:-uu, uu:-uu] = 0

                try:
                    v = (self.imlist[b][i]-self.model_all_rendered[b][i])[np.where(maskarr==1)]
                except:
                    print (b,i)
                    v = (self.imlist[b][i])[np.where(maskarr==1)]

                if len(v) > 50:
                    self.background_subtraction_OK_flag[b][i] = True
                    correction = np.median(v)
                    self.imlist[b][i] -= correction
                    bkg_tot += correction
                    count += 1
                    len_v += len(v)
                else:
                    self.background_subtraction_OK_flag[b][i] = False

        if count == 0:
            self.bkg = bkg_tot
            self.pixel_used_bkg = 1e10
        else:
            self.bkg = bkg_tot / count
            self.pixel_used_bkg = len_v / count            
            
            
            
            
    def zero_padd_psf(self):
        '''
        Zero-pads the PSF solutions so they match the size of the images.
        This is important for ensuring consistency in image processing.
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
                        dx = np.int((size_x-size_psf_x)//2)+1
                        mute[dx:dx+size_psf_x,:][:,dx:dx+size_psf_x] = self.psf[index_band][exp]
                        self.psf[index_band][exp] = mute
            
            

            
            
            
            
        
        
            
            
            
