import pytest
import numpy as np
import sxdes
import bfd
import ngmix.gmix as gmix
import gc
import ngmix
import pyfits as pf
import copy
import ngmix.gmix as gmix
import gc
import ngmix
from ngmix.jacobian import Jacobian
import numpy
import numpy
from ngmix.jacobian.jacobian_nb import jacobian_get_vu, jacobian_get_area
from ngmix.gexceptions import BootPSFFailure, BootGalFailure



NO_ATTEMPT = 2**0
CEN_SHIFT = 2**1
NONPOS_FLUX = 2**2
NONPOS_SIZE = 2**3
LOW_DET = 2**4
MAXITER = 2**5
NONPOS_VAR = 2**6
GMIX_RANGE_ERROR = 2**7
NONPOS_SHAPE_VAR = 2**8

# flags for LM fitting diagnostics
LM_SINGULAR_MATRIX = 2 ** 9
LM_NEG_COV_EIG = 2 ** 10
LM_NEG_COV_DIAG = 2 ** 11
LM_FUNC_NOTFINITE = 2 ** 12

# for LM this indicates a the eigenvalues of the covariance cannot be found
EIG_NOTFINITE = 2 ** 13

DIV_ZERO = 2 ** 14  # division by zero
ZERO_DOF = 2 ** 15  # dof zero so can't do chi^2/dof

# these mappings keep the API the same
EM_RANGE_ERROR = GMIX_RANGE_ERROR
EM_MAXITER = MAXITER
BAD_VAR = NONPOS_VAR

NAME_MAP = {
    # no attempt was made to measure this object, usually
    # due to a previous step in the code fails.
    NO_ATTEMPT: 'no attempt',

    # flag for the center shifting too far
    # used by admom
    CEN_SHIFT: 'center shifted too far',

    NONPOS_FLUX: 'flux <= 0',
    NONPOS_SIZE: 'T <= 0',
    LOW_DET: 'determinant near zero',
    MAXITER: 'max iterations reached',
    NONPOS_VAR: 'non-positive (definite) variance',
    NONPOS_SHAPE_VAR: 'non-positive shape variance',
    GMIX_RANGE_ERROR: 'GMixRangeError raised',

    LM_SINGULAR_MATRIX: 'singular matrix in LM',
    LM_NEG_COV_EIG: 'negative covariance eigenvalue in LM',
    LM_NEG_COV_DIAG: 'negative covariance diagional value in LM',
    LM_FUNC_NOTFINITE: 'function not finite in LM',

    # for LM this indicates a the eigenvalues of the covariance cannot be found
    EIG_NOTFINITE: 'eigenvalues of covariance cannot be found in LM',

    DIV_ZERO: 'divide by zero',
    ZERO_DOF: 'degrees of freedom for it is zero (no chi^2/dof possible)',
}


def get_flags_str(val, name_map=None):
    """Get a descriptive string given a flag value.
    Parameters
    ----------
    val : int
        The flag value.
    name_map : dict, optional
        A dictionary mapping values to names. Default is global at
        ngmix.flags.NAME_MAP.
    Returns
    -------
    flagstr : str
        A string of descriptions for each bit separated by `|`.
    """
    if name_map is None:
        name_map = NAME_MAP

    nstrs = []
    for pow in range(32):
        fval = 2**pow
        if ((val & fval) != 0):
            if fval in name_map:
                nstrs.append(name_map[fval])
            else:
                nstrs.append("bit 2**%d" % pow)
    return "|".join(nstrs)




class MetadataMixin(object):
    @property
    def meta(self):
        """
        get the metadata dictionary
        currently this simply returns a reference
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """
        set the metadata dictionary
        This method does consistency checks and will raise a TypeError if the input is
        not None or a Python dict.
        """
        self.set_meta(meta)

    def set_meta(self, meta):
        """
        set the metadata dictionary
        This method does consistency checks and will raise a TypeError if the input is
        not None or a Python dict.
        """
        if meta is None:
            meta = {}

        if not isinstance(meta, dict):
            raise TypeError("meta data must be in "
                            "dictionary form, got %s" % type(meta))

        self._meta = meta

    def update_meta_data(self, meta):
        """
        Update the metadata dictionary
        This method does consistency checks and will raise a TypeError if the input is
        not a Python dict.
        """
        if not isinstance(meta, dict):
            raise TypeError(
                "meta data must be in dictionary form, got %s" % type(meta)
            )
        self.meta.update(meta)

        
class Observation(MetadataMixin):
    """
    Represent an observation with an image and possibly a
    weight map and jacobian
    parameters
    ----------
    image: ndarray
        The image
    weight: ndarray, optional
        Weight map, same shape as image
    bmask: ndarray, optional
        A bitmask array
    ormask: ndarray, optional
        A bitmask array
    noise: ndarray, optional
        A noise field to associate with this observation
    jacobian: Jacobian, optional
        Type Jacobian or a sub-type
    gmix: GMix, optional
        Optional GMix object associated with this observation
    psf: Observation, optional
        Optional psf Observation
    meta: dict
        Optional dictionary
    mfrac: ndarray, optional
        A masked fraction image for this observation.
    ignore_zero_weight: bool
        If True, do not store zero weight pixels in the pixels
        array.  Default is True.
    store_pixels: bool
        If True, store an array of pixels for use in fitting routines.
        If False, the ignore_zero_weight keyword is not used.
    ignore_zero_weight: bool
        Only relevant if store_pixels is True.
        If ignore_zero_weight is True, then zero-weight pixels are ignored
        when constructing the internal pixels array for fitting routines.
        If False, then zero-weight pixels are included in the internal pixels
        array.
    notes
    -----
    Updates of the internal data of ngmix.Observation will only work in
    a python context, e.g:
        with obs.writeable():
            obs.image[w] += 5
    """
    def __init__(self,
                 image,
                 weight=None,
                 bmask=None,
                 ormask=None,
                 noise=None,
                 jacobian=None,
                 gmix=None,
                 psf=None,
                 meta=None,
                 mfrac=None,
                 store_pixels=True,
                 ignore_zero_weight=True):

        self._writeable = False
        self._ignore_zero_weight = ignore_zero_weight
        self._store_pixels = store_pixels

        # pixels depends on image, weight and jacobian, so delay until all are
        # set

        self.set_image(image, update_pixels=False)

        # If these are None, they get default values

        self.set_weight(weight, update_pixels=False)
        self.set_jacobian(jacobian, update_pixels=False)

        # now image, weight, and jacobian are set, create
        # the pixel array
        self.update_pixels()

        self.set_meta(meta)

        # optional, if None nothing is set
        self.set_bmask(bmask)
        self.set_ormask(ormask)
        self.set_noise(noise)
        self.set_gmix(gmix)
        self.set_psf(psf)
        self.set_mfrac(mfrac)

    @property
    def image(self):
        """
        getter for image
        returns a read-only reference
        """
        return self._get_view(self._image)

    @image.setter
    def image(self, image):
        """
        set the image
        this does consistency checks and can trigger an update
        of the pixels array
        """
        self.set_image(image)

    @property
    def weight(self):
        """
        getter for weight
        returns a read-only reference
        """
        return self._get_view(self._weight)

    @weight.setter
    def weight(self, weight):
        """
        set the weight
        this does consistency checks and can trigger an update
        of the pixels array
        """
        self.set_weight(weight)

    @property
    def pixels(self):
        """
        getter for pixels
        this simply returns a reference.  Note the pixels array is *always*
        read only.  To reset the pixels you must reset the image/weight/jacobian
        """
        return self._pixels

    @property
    def mfrac(self):
        """
        getter for mfrac
        returns a read-only reference
        """
        return self._get_view(self._mfrac)

    @mfrac.setter
    def mfrac(self, mfrac):
        """
        set the mfrac, with consistency checks
        """
        self.set_mfrac(mfrac)

    @property
    def bmask(self):
        """
        getter for bmask
        returns a read-only reference
        """
        return self._get_view(self._bmask)

    @bmask.setter
    def bmask(self, bmask):
        """
        set the bmask, with consistency checks
        """
        self.set_bmask(bmask)

    @property
    def ormask(self):
        """
        getter for ormask
        returns a read-only reference
        """
        return self._get_view(self._ormask)

    @ormask.setter
    def ormask(self, ormask):
        """
        set the ormask
        """
        self.set_ormask(ormask)

    @property
    def noise(self):
        """
        getter for noise
        returns a read-only reference
        """
        return self._get_view(self._noise)

    @noise.setter
    def noise(self, noise):
        """
        set the noise
        """
        self.set_noise(noise)

    @property
    def jacobian(self):
        """
        get a read-only reference to the jacobian.  A new jacobian
        is made with read-only reference to underlying data
        """
        return self.get_jacobian()

    @jacobian.setter
    def jacobian(self, jacobian):
        """
        set the jacobian
        """
        self.set_jacobian(jacobian)

    @property
    def gmix(self):
        """
        get a copy of the gaussian mixture
        """
        return self.get_gmix()

    @gmix.setter
    def gmix(self, gmix):
        """
        set the gmix
        """
        self.set_gmix(gmix)

    @property
    def psf(self):
        """
        getter for psf
        currently this simply returns a reference
        """
        return self._psf

    @psf.setter
    def psf(self, psf):
        """
        set the psf
        """
        self.set_psf(psf)

    def set_image(self, image, update_pixels=True):
        """
        Set the image.  If the image is being reset, must be
        same shape as previous incarnation in order to remain
        consistent
        parameters
        ----------
        image: ndarray (or None)
            The new image.
        update_pixels: bool
            If True, update the internal pixels array. If False, do not
            do this update. Default is True.
        """

        if hasattr(self, '_image'):
            image_old = self._image
        else:
            image_old = None

        # force f8 with native byte ordering, contiguous C layout
        image = np.ascontiguousarray(image, dtype='f8')

        assert len(image.shape) == 2, "image must be 2d"

        if image_old is not None:
            mess = ("old and new image must have same shape, to "
                    "maintain consistency, got %s "
                    "vs %s" % (image.shape, image_old.shape))
            assert image.shape == image_old.shape, mess

        self._image = image

        if update_pixels:
            self.update_pixels()

    def set_weight(self, weight, update_pixels=True):
        """
        Set the weight map.
        parameters
        ----------
        weight: ndarray (or None)
            The new weight image.
        update_pixels: bool
            If True, update the internal pixels array. If False, do not
            do this update. Default is True.
        """

        image = self.image
        if weight is not None:
            # force f8 with native byte ordering, contiguous C layout
            weight = np.ascontiguousarray(weight, dtype='f8')
            assert len(weight.shape) == 2, "weight must be 2d"

            mess = "image and weight must be same shape"
            assert (weight.shape == image.shape), mess

        else:
            weight = np.zeros(image.shape) + 1.0

        self._weight = weight
        if update_pixels:
            self.update_pixels()

    def set_mfrac(self, mfrac):
        """
        Set the masked fraction entry.
        parameters
        ----------
        mfrac: ndarray (or None)
            The new masked fraction image.
        """
        if mfrac is None:
            if self.has_mfrac():
                del self._mfrac
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            mfrac = np.ascontiguousarray(mfrac)
            assert len(mfrac.shape) == 2, "mfrac must be 2d"

            assert (mfrac.shape == image.shape), \
                "image and mfrac must be same shape"

            self._mfrac = mfrac

    def has_mfrac(self):
        """
        returns True if a masked fraction image is set
        """
        if hasattr(self, '_mfrac'):
            return True
        else:
            return False

    def set_bmask(self, bmask):
        """
        Set the bitmask
        parameters
        ----------
        bmask: ndarray (or None)
            The new bit mask image.
        """
        if bmask is None:
            if self.has_bmask():
                del self._bmask
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            bmask = np.ascontiguousarray(bmask)
            assert len(bmask.shape) == 2, "bmask must be 2d"

            assert (bmask.shape == image.shape), \
                "image and bmask must be same shape"

            self._bmask = bmask

    def has_bmask(self):
        """
        returns True if a bitmask is set
        """
        if hasattr(self, '_bmask'):
            return True
        else:
            return False

    def set_ormask(self, ormask):
        """
        Set the bitmask
        parameters
        ----------
        ormask: ndarray (or None)
            The new "or" mask image.
        """
        if ormask is None:
            if self.has_ormask():
                del self._ormask
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            ormask = np.ascontiguousarray(ormask)
            assert len(ormask.shape) == 2, "ormask must be 2d"

            assert (ormask.shape == image.shape),\
                "image and ormask must be same shape"

            self._ormask = ormask

    def has_ormask(self):
        """
        returns True if a bitmask is set
        """
        if hasattr(self, '_ormask'):
            return True
        else:
            return False

    def set_noise(self, noise):
        """
        Set a noise image
        parameters
        ----------
        noise: ndarray (or None)
            The new noise image.
        """
        if noise is None:
            if self.has_noise():
                del self._noise
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            noise = np.ascontiguousarray(noise)
            assert len(noise.shape) == 2, "noise must be 2d"

            assert (noise.shape == image.shape), \
                "image and noise must be same shape"

            self._noise = noise

    def has_noise(self):
        """
        returns True if a bitmask is set
        """
        if hasattr(self, '_noise'):
            return True
        else:
            return False

    def set_jacobian(self, jacobian, update_pixels=True):
        """
        Set the jacobian.  If None is sent, a UnitJacobian is generated with
        center equal to the canonical center
        parameters
        ----------
        jacobian: Jacobian (or None)
            The new jacobian.
        update_pixels: bool
            If True, update the internal pixels array. If False, do not
            do this update. Default is True.
        """
        if jacobian is None:
            cen = (np.array(self.image.shape)-1.0)/2.0
            jac = UnitJacobian(row=cen[0], col=cen[1])
        else:
            mess = ("jacobian must be of "
                    "type Jacobian, got %s" % type(jacobian))
            assert isinstance(jacobian, Jacobian), mess
            jac = jacobian.copy()

        self._jacobian = jac

        if update_pixels:
            self.update_pixels()

    def get_jacobian(self):
        """
        get a jacobian with reference to our jacobian's data
        this is not writeable by default
        """
        j = self._jacobian.copy()
        j._data = self._get_view(self._jacobian._data)
        return j

    def set_psf(self, psf):
        """
        Set a psf Observation
        """
        if self.has_psf():
            del self._psf

        if psf is not None:
            mess = "psf must be of Observation, got %s" % type(psf)
            assert isinstance(psf, Observation), mess
            self._psf = psf

    def get_psf(self):
        """
        get the psf object
        """
        if not self.has_psf():
            raise RuntimeError("this obs has no psf set")
        return self._psf

    def has_psf(self):
        """
        does this object have a psf set?
        """
        return hasattr(self, '_psf')

    def get_psf_gmix(self):
        """
        get the psf gmix if it exists
        """
        if not self.has_psf_gmix():
            raise RuntimeError("this obs has not psf set with a gmix")
        return self.psf.get_gmix()

    def has_psf_gmix(self):
        """
        does this object have a psf set, which has a gmix set?
        """
        if self.has_psf():
            return self.psf.has_gmix()
        else:
            return False

    def set_gmix(self, gmix):
        """
        Set the gmix.
        parameters
        ----------
        gmix: ngmix.GMix
            The GMix to use to set the internal GMix.
        """

        if self.has_gmix():
            del self._gmix

        if gmix is not None:
            mess = "gmix must be of type GMix, got %s" % type(gmix)
            assert isinstance(gmix, GMix), mess
            self._gmix = gmix.copy()

    def get_gmix(self):
        """
        get a copy of the gmix object
        """
        if not self.has_gmix():
            raise RuntimeError("this obs has not gmix set")
        return self._gmix.copy()

    def has_gmix(self):
        """
        does this object have a gmix set?
        """
        return hasattr(self, '_gmix')

    def get_s2n(self):
        """
        get the the simple s/n estimator
        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
        returns
        -------
        s2n: float
            The s/n of the images. Will retun -9999 if the s/n cannot be computed.
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/np.sqrt(Vsum)
        else:
            s2n = -9999.0
        return s2n

    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator
        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
        returns
        -------
        Isum: float
            The value sum(I).
        Vsum: float
            The value sum(1/w)
        Npix: int
            The number of non-zero-weight pixels.
        """

        image = self.image
        weight = self.weight

        w = np.where(weight > 0)

        if w[0].size > 0:
            Isum = image[w].sum()
            Vsum = (1.0/weight[w]).sum()
            Npix = w[0].size
        else:
            Isum = 0.0
            Vsum = 0.0
            Npix = 0

        return Isum, Vsum, Npix

    def copy(self):
        """
        make a copy of the observation
        """
        if self.has_bmask():
            bmask = self.bmask.copy()
        else:
            bmask = None

        if self.has_ormask():
            ormask = self.ormask.copy()
        else:
            ormask = None

        if self.has_noise():
            noise = self.noise.copy()
        else:
            noise = None

        if self.has_gmix():
            # makes a copy
            gmix = self.gmix
        else:
            gmix = None

        if self.has_psf():
            psf = self.psf.copy()
        else:
            psf = None

        if self.has_mfrac():
            mfrac = self.mfrac.copy()
        else:
            mfrac = None

        meta = copy.deepcopy(self.meta)

        return Observation(
            self.image.copy(),
            weight=self.weight.copy(),
            bmask=bmask,
            ormask=ormask,
            noise=noise,
            gmix=gmix,
            jacobian=self.jacobian,  # makes a copy internally
            meta=meta,
            psf=psf,
            mfrac=mfrac,
        )

    def update_pixels(self):
        """
        create the pixel struct array, for efficient cache usage
        """

        if not self._store_pixels:
            self._pixels = None
            return

        pixels = make_pixels(
            self.image,
            self.weight,
            self._jacobian,
            ignore_zero_weight=self._ignore_zero_weight,
        )
        pixels.flags['WRITEABLE'] = False
        self._pixels = pixels

    def _get_view(self, data):
        """return a view of some numpy data.
        The `_writeable` attribute is set by the context management methods
        `__enter__` and `__exit__` so that the internal data of the Observation
        can only be updated in place within a context manager.
        """
        view = data.view()
        view.flags['WRITEABLE'] = self._writeable
        return view

    def writeable(self):
        """
        returns self
        This method is meant to be used when updating the data of an Observation,
        e.g.,
            with obs.writeable():
                obs.image[w] += 5
        """
        return self

    def __enter__(self):
        self._writeable = True
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._writeable = False
        self.update_pixels()

        

def make_pixels(image, weight, jacob, ignore_zero_weight=True):
    """
    make a pixel array from the image and weight
    stores v,u image value, and 1/err for each pixel
    parameters
    ----------
    pixels: array
        1-d array of pixel structures, u,v,val,ierr
    image: 2-d array
        2-d image array
    weight: 2-d array
        2-d image array same shape as image
    jacob: jacobian structure
        row0,col0,dvdrow,dvdcol,dudrow,dudcol,...
    ignore_zero_weight: bool
        If set, zero or negative weight pixels are ignored.  In this case the
        returned pixels array is equal in length to the set of positive weight
        pixels in the weight image.  Default True.
    returns
    -------
    1-d pixels array
    """
    #from .pixels_nb import fill_pixels

    if ignore_zero_weight:
        w = numpy.where(weight > 0.0)
        if w[0].size == 0:
            raise GMixFatalError("no weights > 0")
        npixels = w[0].size
    else:
        npixels = image.size

    pixels = numpy.zeros(npixels, dtype=_pixels_dtype)

    fill_pixels(
        pixels,
        image,
        weight,
        jacob._data,
        ignore_zero_weight=ignore_zero_weight,
    )

    return pixels


def make_coords(dims, jacob):
    """
    make a coords array
    """
   # from .pixels_nb import fill_coords

    nrow, ncol = dims

    coords = numpy.zeros(nrow * ncol, dtype=_coords_dtype)

    fill_coords(
        coords, nrow, ncol, jacob._data,
    )

    return coords


_pixels_dtype = [
    ("u", "f8"),
    ("v", "f8"),
    ("area", "f8"),
    ("val", "f8"),
    ("ierr", "f8"),
    ("fdiff", "f8"),
]


_coords_dtype = [
    ("u", "f8"),
    ("v", "f8"),
    ("area", "f8"),
]




def fill_pixels(pixels, image, weight, jacob, ignore_zero_weight=True):
    """
    store v,u image value, and 1/err for each pixel
    store into 1-d pixels array
    parameters
    ----------
    pixels: array
        1-d array of pixel structures, u,v,val,ierr
    image: 2-d array
        2-d image array
    weight: 2-d array
        2-d image array same shape as image
    jacob: jacobian structure
        row0,col0,dvdrow,dvdcol,dudrow,dudcol,...
    ignore_zero_weight: bool
        If set, zero or negative weight pixels are ignored.
        In this case it verified that the input pixels
        are equal in length to the set of positive weight
        pixels in the weight image.  Default True.
    """
    nrow, ncol = image.shape
    pixel_area = jacobian_get_area(jacob)

    ipixel = 0
    for row in range(nrow):
        for col in range(ncol):

            ivar = weight[row, col]
            if ignore_zero_weight and ivar <= 0.0:
                continue

            pixel = pixels[ipixel]

            v, u = jacobian_get_vu(jacob, row, col)

            pixel['v'] = v
            pixel['u'] = u
            pixel['area'] = pixel_area

            pixel['val'] = image[row, col]

            if ivar < 0.0:
                ivar = 0.0

            pixel['ierr'] = numpy.sqrt(ivar)

            ipixel += 1

    if ipixel != pixels.size:
        raise RuntimeError('some pixels were not filled')



def fill_coords(coords, nrow, ncol, jacob):
    """
    store v,u image value, and 1/err for each pixel
    store into 1-d pixels array
    parameters
    ----------
    pixels: array
        1-d array of pixel structures, u,v,val,ierr
    image: 2-d array
        2-d image array
    weight: 2-d array
        2-d image array same shape as image
    jacob: jacobian structure
        row0,col0,dvdrow,dvdcol,dudrow,dudcol,...
    """

    pixel_area = jacobian_get_area(jacob)

    icoord = 0
    for row in range(nrow):
        for col in range(ncol):

            coord = coords[icoord]

            v, u = jacobian_get_vu(jacob, row, col)

            coord['v'] = v
            coord['u'] = u
            coord['area'] = pixel_area

            icoord += 1

#import eastlake
import sys
sys.path.append('/global/u2/m/mgatti/eastlake/eastlake/steps/newish_metacal/')
sys.path.append('/global/u2/m/mgatti/eastlake/eastlake/steps/newish_metacal/metacal/')

DEFAULT_MAXITER = 200
DEFAULT_SHIFTMAX = 5.0  # pixels
DEFAULT_ETOL = 1.0e-5
DEFAULT_TTOL = 1.0e-3

#from ngmix.admom import AdmomFitter


__all__ = ['Fitter', 'CoellipFitter', 'PSFFluxFitter']
#import logging

#from .leastsqbound import run_leastsq
#from .. import gmix
#from ..defaults import DEFAULT_LM_PARS
#from .results import FitModel, CoellipFitModel, PSFFluxFitModel

#LOGGER = logging.getLogger(__name__)


class Fitter(object):
    """
    A class for doing a fit using levenberg marquardt
    Parameters
    ----------
    model: str
        The model to fit
    prior: ngmix prior
        A prior for fitting
    fit_pars: dict
        Parameters to send to the leastsq fitting routine
    """

    def __init__(self, model, prior=None, fit_pars=None):
        self.prior = prior
        self.model = gmix.get_model_num(model)
        self.model_name = gmix.get_model_name(self.model)

        if fit_pars is not None:
            self.fit_pars = fit_pars.copy()
        else:
            self.fit_pars = DEFAULT_LM_PARS.copy()

    def go(self, obs, guess):
        """
        Run leastsq and set the result
        Parameters
        ----------
        obs: Observation, ObsList, or MultiBandObsList
            Observation(s) to fit
        guess: array
            Array of initial parameters for the fit
        Returns
        --------
        a dict-like which contains the result as well as functions used for the
        fitting.
        """

        fit_model = self._make_fit_model(obs=obs, guess=guess)

        result = run_leastsq(
            fit_model.calc_fdiff,
            guess=guess,
            n_prior_pars=fit_model.n_prior_pars,
            bounds=fit_model.bounds,
            **self.fit_pars
        )

        fit_model.set_fit_result(result)
        return fit_model

    def _make_fit_model(self, obs, guess):
        return FitModel(
            obs=obs, model=self.model, guess=guess, prior=self.prior,
        )


class CoellipFitter(Fitter):
    """
    class to perform a fit using a model of coelliptical gaussians
    Parameters
    ----------
    ngauss: int
        The number of coelliptical gaussians to fit
    prior: ngmix prior
        A prior for fitting
    fit_pars: dict
        Parameters to send to the leastsq fitting routine
    """

    def __init__(self, ngauss, prior=None, fit_pars=None):
        self._ngauss = ngauss
        super().__init__(model="coellip", prior=prior, fit_pars=fit_pars)

    def _make_fit_model(self, obs, guess):
        return CoellipFitModel(
            obs=obs, ngauss=self._ngauss, guess=guess, prior=self.prior,
        )


class PSFFluxFitter(object):
    """
    Calculate a psf flux or template flux.  We fix the center, so this is
    linear.  This uses a simple cross-correlation between model and data.
    The center of the jacobian(s) must point to a common place on the sky, and
    if the center is input (to reset the gmix centers),) it is relative to that
    position
    Parameters
    -----------
    do_psf: bool, optional
        If True, use the gaussian mixtures in the psf observation as templates.
        In this mode the code calculates a "psf flux".  If set for False,
        templates are taken from the primary observations. Default True.
    normalize_psf: True or False
        if True, then normalize PSF gmix to flux of unity, otherwise use input
        normalization.  Default True
    """

    def __init__(self, do_psf=True, normalize_psf=True):
        self.do_psf = do_psf
        self.normalize_psf = normalize_psf

    def go(self, obs):
        """
        perform the template flux fit and return the result
        Returns
        --------
        a dict-like which contains the result as well as functions used for the
        fitting. The class is TemplateFluxFitModel
        """
        fit_model = PSFFluxFitModel(
            obs=obs, do_psf=self.do_psf, normalize_psf=self.normalize_psf,
        )
        fit_model.go()
        return fit_model



class RunnerBase(object):
    """
    Run a fitter and guesser on observations
    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters.
    ntry: int, optional
        Number of times to try if there is failure
    """
    def __init__(self, fitter, guesser=None, ntry=1):
        self.fitter = fitter
        self.guesser = guesser
        self.ntry = ntry


class Runner(RunnerBase):
    """
    Run a fitter and guesser on observations
    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters.
    ntry: int, optional
        Number of times to try if there is failure
    """
    def go(self, obs):
        """
        Run the fitter on the input observation(s), possibly multiple times
        using guesses generated from the guesser
        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        Returns
        -------
        result dictionary
        """

        return run_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
        )


class PSFRunner(RunnerBase):
    """
    Run a fitter on each psf observation.
    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    set_result: bool
        If True, set the result and possibly a gmix in the observation.
        Default True
    """
    def __init__(self, fitter, guesser=None, ntry=1, set_result=True):
        self.fitter = fitter
        self.guesser = guesser
        self.ntry = ntry
        self.set_result = set_result

    def go(self, obs):
        """
        Run the fitter on the psf observations associated with the input
        observation(s), possibly multiple times using guesses generated from
        the guesser.
        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        ntry: int, optional
            Number of times to try if there is failure
        Returns
        --------
        result if obs is an Observation
        result list if obs is an ObsList
        list of result lists if obs is a MultiBandObsList
        Side Effects
        ------------
        if set_result is True, then obs.meta['result'] is set to the fit result
        and the .gmix attribuite is set for each successful fit, if appropriate
        """

        return run_psf_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
            set_result=self.set_result,
        )


def run_fitter(obs, fitter, guesser=None, ntry=1):
    """
    run a fitter multiple times if needed, with guesses generated from the
    input guesser
    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    Returns
    -------
    result dictionary
    """

    for i in range(ntry):

        if guesser is not None:
            guess = guesser(obs=obs)
            res = fitter.go(obs=obs, guess=guess)
        else:
            res = fitter.go(obs=obs)

        if res['flags'] == 0:
            break

    return res


def run_psf_fitter(obs, fitter, guesser=None, ntry=1, set_result=True):
    """
    run a fitter on each observation in the input observation(s).  The fitter
    will be run multiple times if needed, with guesses generated from the input
    guesser if one is sent.  If a psf obs is set that is fit rather than the
    primary observation.
    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    set_result: bool
        If True, set the result and possibly a gmix in the observation.
        Default True
    Side Effects
    ------------
    if set_result is True, then obs.meta['result'] is set to the fit result
    and the .gmix attribuite is set for each successful fit, if appropriate
    """

    if isinstance(obs, MultiBandObsList):
        reslol = []
        for tobslist in obs:
            reslist = run_psf_fitter(
                obs=tobslist, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=set_result,
            )
            reslol.append(reslist)
        return reslol

    elif isinstance(obs, ObsList):
        reslist = []
        for tobs in obs:
            res = run_psf_fitter(
                obs=tobs, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=set_result,
            )
            reslist.append(res)
        return reslist

    elif isinstance(obs, Observation):

        if obs.has_psf():
            obs_to_fit = obs.psf
        else:
            obs_to_fit = obs

        res = run_fitter(
            obs=obs_to_fit, fitter=fitter, guesser=guesser, ntry=ntry,
        )

        if set_result:
            obs_to_fit.meta['result'] = res

            if res['flags'] == 0 and hasattr(res, 'get_gmix'):
                gmix = res.get_gmix()
                obs_to_fit.gmix = gmix

        return res

    else:
        raise ValueError(
            'obs must be an Observation, ObsList, or MultiBandObsList'
        )
       
class TFluxGuesser(object):
    """
    get full guesses from just T, fluxes
    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    T: float
        Central value for T guesses
    flux: float or sequence
        Central value for flux guesses.  Can be a single float or a sequence/array
        for multiple bands
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, T, flux, prior=None):
        self.rng = rng
        self.T = T

        self.fluxes = np.array(flux, dtype="f8", ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess.  The center, shape are distributed tightly around
        zero
        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        npars = 5 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 1] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 2] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 3] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 4] = self.T * rng.uniform(low=0.9, high=1.1, size=nrand)

        for band in range(nband):
            guess[:, 5 + band] = (
                fluxes[band] * rng.uniform(low=0.9, high=1.1, size=nrand)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]

        return guess


class TPSFFluxGuesser(object):
    """
    get full guesses from just the input T and fluxes based on psf fluxes
    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    T: float
        Central value for T guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, T, prior=None):
        self.rng = rng
        self.T = T
        self.prior = prior
        self._id_last = None
        self._psf_fluxes = None

    def _get_psf_fluxes(self, obs):
        oid = id(obs)
        if oid != self._id_last:
            self._id_last = oid
            fdict = _get_psf_fluxes(rng=self.rng, obs=obs)
            self._psf_fluxes = fdict['flux']
        return self._psf_fluxes

    def __call__(self, obs, nrand=1):
        """
        Generate a guess.
        Parameters
        ----------
        obs: Observation
            The observation(s) used for psf fluxes
        nrand: int, optional
            Number of samples to draw.  Default 1
        """

        rng = self.rng

        fluxes = self._get_psf_fluxes(obs=obs)

        nband = fluxes.size
        npars = 5 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 1] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 2] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 3] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 4] = self.T * rng.uniform(low=0.9, high=1.1, size=nrand)

        for band in range(nband):
            guess[:, 5 + band] = (
                fluxes[band] * rng.uniform(low=0.9, high=1.1, size=nrand)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]

        return guess


class TPSFFluxAndPriorGuesser(TPSFFluxGuesser):
    """
    get full guesses from just the T guess, psf fluxes and the prior
    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    T: float
        Central value for T guesses
    prior: joint prior
        cen, g drawn from this prior
    """

    def __init__(self, rng, T, prior):
        self.rng = rng
        self.T = T
        self.prior = prior
        self._id_last = None
        self._psf_fluxes = None

    def __call__(self, obs, nrand=1):
        """
        Generate a guess.
        Parameters
        ----------
        obs: Observation
            The observation(s) used for psf fluxes
        nrand: int, optional
            Number of samples to draw.  Default 1
        """

        rng = self.rng

        fluxes = self._get_psf_fluxes(obs=obs)

        nband = fluxes.size

        guess = self.prior.sample(nrand)

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        for band in range(nband):
            guess[:, 5 + band] = (
                fluxes[band] * rng.uniform(low=0.9, high=1.1, size=nrand)
            )

        if self.prior is not None:
            _fix_guess_TFlux(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]

        return guess


def _get_psf_fluxes(rng, obs):
    """
    Get psf fluxes for the input observations
    The result is cached with cache size 64
    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    obs: Observation, Obslist, MultiBandObsList
        The observations
    Returns
    -------
    a dict with 'flags' 'flux' 'flux_err' for each band
    """
    from .fitting import PSFFluxFitter
    from .observation import get_mb_obs

    mbobs = get_mb_obs(obs)

    nband = len(mbobs)
    flux = np.zeros(nband)
    flux_err = np.zeros(nband)
    flags = np.zeros(nband, dtype='i4')

    fitter = PSFFluxFitter()

    for iband, obslist in enumerate(mbobs):
        res = fitter.go(obs=obslist)

        # flags are set for all zero weight pixels, so there isn't anything we
        # can do, we'll have to fix it up
        flags[iband] = res['flags']
        flux[iband] = res['flux']
        flux_err[iband] = res['flux_err']

    logic = (flags == 0) & np.isfinite(flux)
    wgood, = np.where(logic)
    if wgood.size != nband:
        if wgood.size == 0:
            # no good fluxes due to flags. This means we have no information
            # to use for a fit or measurement, so there is no point in
            # generating a guess
            raise PSFFluxFailure("no good psf fluxes")
        else:
            # make a guess based on the good ones
            wbad, = np.where(~logic)
            fac = 1.0 + rng.uniform(low=-0.1, high=0.1, size=wbad.size)
            flux[wbad] = flux[wgood].mean()*fac

    return {
        'flags': flags,
        'flux': flux,
        'flux_err': flux_err,
    }


class TFluxAndPriorGuesser(object):
    """
    Make guesses from the input T, flux and prior
    parameters
    ----------
    T: float
        Center for T guesses
    flux: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    """

    def __init__(self, rng, T, flux, prior):

        fluxes = np.array(flux, dtype="f8", ndmin=1)

        self.T = T
        self.fluxes = fluxes
        self.prior = prior

        lfluxes = self.fluxes.copy()
        (w,) = np.where(self.fluxes < 0.0)
        if w.size > 0:
            lfluxes[w[:]] = 1.0e-10

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess, with center and shape drawn from the prior.
        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        nband = fluxes.size

        guess = self.prior.sample(nrand)

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=nrand)
            guess[:, 5 + band] = fluxes[band] * (1.0 + r)

        _fix_guess_TFlux(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class BDFGuesser(object):
    """
    Make BDF guesses from the input T, flux and prior
    parameters
    ----------
    T: float
        Center for T guesses
    flux: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    """

    def __init__(self, T, flux, prior):
        self.T = T
        self.fluxes = np.array(flux, ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        center, shape are just distributed around zero
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """
        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        guess = self.prior.sample(nrand)

        nband = fluxes.size

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        # fracdev prior
        guess[:, 5] = rng.uniform(low=0.4, high=0.6, size=nrand)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=nrand)
            guess[:, 6 + band] = fluxes[band] * (1.0 + r)

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class BDGuesser(object):
    """
    Make BD guesses from the input T, flux and prior
    parameters
    ----------
    T: float
        Center for T guesses
    flux: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    """

    def __init__(self, T, flux, prior):
        self.T = T
        self.fluxes = np.array(flux, ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess from the T, flux and prior for the rest
        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        guess = self.prior.sample(nrand)

        nband = fluxes.size

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        # fracdev prior
        guess[:, 5] = rng.uniform(low=0.4, high=0.6, size=nrand)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=nrand)
            guess[:, 7 + band] = fluxes[band] * (1.0 + r)

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class ParsGuesser(object):
    """
    Make guess based on an input set of parameters
    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    pars: array/sequence
        Parameters upon which to base the guess
    widths: array/sequence
        Widths for random guesses, default is 0.1, absolute
        for cen, g but relative for T, flux
    prior: joint prior, optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, pars, prior=None, widths=None):
        self.rng = rng
        self.pars = np.array(pars)
        self.prior = prior

        self.np = self.pars.size

        if widths is None:
            self.widths = self.pars * 0 + 0.1
            self.widths[0:0+2] = 0.02
        else:
            self.widths = widths

    def __call__(self, nrand=None, obs=None):
        """
        Generate a guess
        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        pars = self.pars
        widths = self.widths

        guess = np.zeros((nrand, self.np))
        guess[:, 0] = pars[0] + widths[0] * srandu(nrand, rng=rng)
        guess[:, 1] = pars[1] + widths[1] * srandu(nrand, rng=rng)

        # prevent from getting too large
        guess_shape = get_shape_guess(
            rng=rng,
            g1=pars[2],
            g2=pars[3],
            nrand=nrand,
            width=widths[2:2+2],
            max=0.8
        )
        guess[:, 2] = guess_shape[:, 0]
        guess[:, 3] = guess_shape[:, 1]

        for i in range(4, self.np):
            guess[:, i] = pars[i] * (1.0 + widths[i] * srandu(nrand, rng=rng))

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if is_scalar:
            guess = guess[0, :]

        return guess


def get_shape_guess(rng, g1, g2, nrand, width, max=0.99):
    """
    Get guess, making sure the range is OK
    """

    g = np.sqrt(g1 ** 2 + g2 ** 2)
    if g > max:
        fac = max / g

        g1 = g1 * fac
        g2 = g2 * fac

    guess = np.zeros((nrand, 2))
    shape = Shape(g1, g2)

    for i in range(nrand):

        while True:
            try:
                g1_offset = width[0] * srandu(rng=rng)
                g2_offset = width[1] * srandu(rng=rng)
                shape_new = shape.get_sheared(g1_offset, g2_offset)
                break
            except GMixRangeError:
                pass

        guess[i, 0] = shape_new.g1
        guess[i, 1] = shape_new.g2

    return guess


class R50FluxGuesser(object):
    """
    get full guesses from just r50 and fluxes
    parameters
    ----------
    r50: float
        Center for r50 (half light radius )guesses
    flux: float or sequences
        Center for flux guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, r50, flux, prior=None):

        self.rng = rng
        if r50 < 0.0:
            raise GMixRangeError("r50 <= 0: %g" % r50)

        self.r50 = r50

        self.fluxes = np.array(flux, dtype="f8", ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess.
        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        npars = 5 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 1] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 2] = 0.02 * srandu(nrand, rng=rng)
        guess[:, 3] = 0.02 * srandu(nrand, rng=rng)

        guess[:, 4] = self.r50 * (1.0 + 0.1 * srandu(nrand, rng=rng))

        fluxes = self.fluxes
        for band in range(nband):
            guess[:, 5 + band] = fluxes[band] * (
                1.0 + 0.1 * srandu(nrand, rng=rng)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class PriorGuesser(object):
    """
    get guesses simply sampling from a prior
    Parameters
    ----------
    prior: joint prior
        A joint prior for all parameters
    """
    def __init__(self, prior):
        self.prior = prior

    def __call__(self, obs=None, nrand=None):
        """
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """
        return self.prior.sample(nrand)


class R50NuFluxGuesser(R50FluxGuesser):
    """
    get full guesses from just r50 spergel nu and fluxes
    parameters
    ----------
    r50: float
        Center for r50 (half light radius )guesses
    nu: float
        Index for the spergel function
    flux: float or sequences
        Center for flux guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    NUMIN = -0.99
    NUMAX = 3.5

    def __init__(self, rng, r50, nu, flux, prior=None):
        super(R50NuFluxGuesser, self).__init__(
            rng=rng, r50=r50, flux=flux, prior=prior,
        )

        if nu < self.NUMIN:
            nu = self.NUMIN
        elif nu > self.NUMAX:
            nu = self.NUMAX

        self.nu = nu

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess
        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        npars = 6 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 1] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 2] = 0.02 * srandu(nrand, rng=rng)
        guess[:, 3] = 0.02 * srandu(nrand, rng=rng)

        guess[:, 4] = self.r50 * (1.0 + 0.1 * srandu(nrand, rng=rng))

        for i in range(nrand):
            while True:
                nuguess = self.nu * (1.0 + 0.1 * srandu(rng=rng))
                if nuguess > self.NUMIN and nuguess < self.NUMAX:
                    break
            guess[i, 5] = nuguess

        fluxes = self.fluxes
        for band in range(nband):
            guess[:, 6 + band] = fluxes[band] * (
                1.0 + 0.1 * srandu(nrand, rng=rng)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class GMixPSFGuesser(object):
    """
    Generate a full gaussian mixture for a psf fit.  Useful for EM and admom
    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    ngauss: int
        number of gaussians
    guess_from_moms: bool, optional
        If set to True, use weighted moments to generate the starting flux and
        T for the guess.  If set to False, the starting flux is gotten from
        summing the image and the fwhm of the guess is set to 3.5 times the
        pixel scale
    """
    def __init__(self, rng, ngauss, guess_from_moms=False):

        self.rng = rng
        self.ngauss = ngauss
        self.guess_from_moms = guess_from_moms

        if self.ngauss == 1:
            self._guess_func = self._get_guess_1gauss
        elif self.ngauss == 2:
            self._guess_func = self._get_guess_2gauss
        elif self.ngauss == 3:
            self._guess_func = self._get_guess_3gauss
        elif self.ngauss == 4:
            self._guess_func = self._get_guess_4gauss
        elif self.ngauss == 5:
            self._guess_func = self._get_guess_5gauss
        else:
            raise ValueError("bad ngauss: %d" % self.ngauss)

    def __call__(self, obs):
        """
        Get a guess for the EM algorithm
        Parameters
        ----------
        obs: Observation
            Starting flux and T for the overall mixture are derived from the
            input observation.  How depends on the gauss_from_moms constructor
            argument
        Returns
        -------
        ngmix.GMix
            The guess mixture, with the number of gaussians as specified in the
            constructor
        """
        return self._get_guess(obs=obs)

    def _get_guess(self, obs):
        T, flux = self._get_T_flux(obs=obs)
        return self._guess_func(flux=flux, T=T)

    def _get_T_flux(self, obs):
        if self.guess_from_moms:
            T, flux = self._get_T_flux_from_moms(obs=obs)
        else:
            T, flux = self._get_T_flux_default(obs=obs)

        return T, flux

    def _get_T_flux_default(self, obs):
        """
        get starting T and flux from a multiple of the pixel scale and the sum
        of the image
        """
        scale = obs.jacobian.scale
        flux = obs.image.sum()
        # for DES 0.9/0.263 = 3.42
        fwhm = scale * 3.5
        T = ngmix.moments.fwhm_to_T(fwhm)
        return T, flux

    def _get_T_flux_from_moms(self, obs):
        """
        get starting T and flux from weighted moments, deweighted
        if the measured T is too small, fall back to the _get_T_flux method
        """
        scale = obs.jacobian.scale

        # 0.9/0.263 = 3.42
        fwhm = scale * 3.5

        Tweight = ngmix.moments.fwhm_to_T(fwhm)
        wt = GMixModel([0.0, 0.0, 0.0, 0.0, Tweight, 1.0], "gauss")

        res = wt.get_weighted_moments(obs=obs, maxrad=1.0e9)
        if res['flags'] != 0:
            return self._get_T_flux_default(obs=obs)

        # ngmix is in flux units, need to divide by area
        area = scale**2

        Tmeas = res['T']

        fwhm_meas = ngmix.moments.T_to_fwhm(Tmeas)
        if fwhm_meas < scale:
            # something probably went wrong
            T, flux = self._get_T_flux(obs=obs)
        else:
            # deweight assuming true profile is a gaussian
            T = 1.0/(1/Tmeas - 1/Tweight)
            flux = res['flux'] * np.pi * (Tweight + T) / area

        return T, flux

    def _get_guess_1gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * rng.uniform(low=0.9, high=1.1),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.2 * sigma2, high=0.2 * sigma2),
            sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_2gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            _em2_pguess[0] * flux,
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em2_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            0.0,
            _em2_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            _em2_pguess[1] * flux,
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em2_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            0.0,
            _em2_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_3gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * _em3_pguess[0] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em3_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em3_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em3_pguess[1] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em3_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em3_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em3_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em3_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em3_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_4gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * _em4_pguess[0] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em4_pguess[1] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em4_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em4_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_5gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * _em5_pguess[0] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[1] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)


_em2_pguess = np.array([0.596510042804182, 0.4034898268889178])
_em2_fguess = np.array([0.5793612389470884, 1.621860687127999])

_em3_pguess = np.array(
    [0.596510042804182, 0.4034898268889178, 1.303069003078001e-07]
)
_em3_fguess = np.array([0.5793612389470884, 1.621860687127999, 7.019347162356363])

_em4_pguess = np.array(
    [0.596510042804182, 0.4034898268889178, 1.303069003078001e-07, 1.0e-8]
)
_em4_fguess = np.array(
    [0.5793612389470884, 1.621860687127999, 7.019347162356363, 16.0]
)

_em5_pguess = np.array(
    [0.59453032, 0.35671819, 0.03567182, 0.01189061, 0.00118906]
)
_em5_fguess = np.array([0.5, 1.0, 3.0, 10.0, 20.0])


class SimplePSFGuesser(GMixPSFGuesser):
    """
    guesser for simple psf fitting
    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    guess_from_moms: bool, optional
        If set to True, use weighted moments to generate the starting flux and
        T for the guess.  If set to False, the starting flux is gotten from
        summing the image and the fwhm of the guess isset to 3.5 times the
        pixel scale
    """
    def __init__(self, rng, guess_from_moms=False):

        self.rng = rng
        self.guess_from_moms = guess_from_moms
        self.npars = 6

    def __call__(self, obs):
        """
        Get a guess for the simple psf
        Parameters
        ----------
        obs: Observation
            Starting flux and T for the overall mixture are derived from the
            input observation.  How depends on the gauss_from_moms constructor
            argument
        Returns
        -------
        guess: array
            The guess array [cen1, cen2, g1, g2, T, flux]
        """
        return self._get_guess(obs=obs)

    def _get_guess(self, obs):
        rng = self.rng
        T, flux = self._get_T_flux(obs=obs)

        guess = np.zeros(self.npars)

        guess[0:0 + 2] += rng.uniform(low=-0.01, high=0.01, size=2)
        guess[2:2 + 2] += rng.uniform(low=-0.05, high=0.05, size=2)

        guess[4] = T * rng.uniform(low=0.9, high=1.1)
        guess[5] = flux * rng.uniform(low=0.9, high=1.1)
        return guess


class CoellipPSFGuesser(GMixPSFGuesser):
    """
    guesser for coelliptical psf fitting
    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    ngauss: int
        number of gaussians
    guess_from_moms: bool, optional
        If set to True, use weighted moments to generate the starting flux and
        T for the guess.  If set to False, the starting flux is gotten from
        summing the image and the fwhm of the guess isset to 3.5 times the
        pixel scale
    """
    def __init__(self, rng, ngauss, guess_from_moms=False):
        super().__init__(
            rng=rng,
            ngauss=ngauss,
            guess_from_moms=guess_from_moms,
        )
        self.npars = get_coellip_npars(ngauss)

    def __call__(self, obs):
        """
        Get a guess for the EM algorithm
        Parameters
        ----------
        obs: Observation
            Starting flux and T for the overall mixture are derived from the
            input observation.  How depends on the gauss_from_moms constructor
            argument
        Returns
        -------
        guess: array
            The guess array, [cen1, cen2, g1, g2, T1, T2, ..., F1, F2, ...]
        """
        return self._get_guess(obs=obs)

    def _make_guess_array(self):
        rng = self.rng
        guess = np.zeros(self.npars)

        guess[0:0 + 2] += rng.uniform(low=-0.01, high=0.01, size=2)
        guess[2:2 + 2] += rng.uniform(low=-0.05, high=0.05, size=2)
        return guess

    def _get_guess_1gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        guess[4] = T * rng.uniform(low=0.9, high=1.1)
        guess[5] = flux * rng.uniform(low=0.9, high=1.1)
        return guess

    def _get_guess_2gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat2_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat2_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = flux * _moffat2_pguess[0] * rng.uniform(low=low, high=high)
        guess[7] = flux * _moffat2_pguess[1] * rng.uniform(low=low, high=high)

        return guess

    def _get_guess_3gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat3_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat3_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = T * _moffat3_fguess[2] * rng.uniform(low=low, high=high)

        guess[7] = flux * _moffat3_pguess[0] * rng.uniform(low=low, high=high)
        guess[8] = flux * _moffat3_pguess[1] * rng.uniform(low=low, high=high)
        guess[9] = flux * _moffat3_pguess[2] * rng.uniform(low=low, high=high)
        return guess

    def _get_guess_4gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat4_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat4_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = T * _moffat4_fguess[2] * rng.uniform(low=low, high=high)
        guess[7] = T * _moffat4_fguess[3] * rng.uniform(low=low, high=high)

        guess[8] = flux * _moffat4_pguess[0] * rng.uniform(low=low, high=high)
        guess[9] = flux * _moffat4_pguess[1] * rng.uniform(low=low, high=high)
        guess[10] = flux * _moffat4_pguess[2] * rng.uniform(low=low, high=high)
        guess[11] = flux * _moffat4_pguess[3] * rng.uniform(low=low, high=high)
        return guess

    def _get_guess_5gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat5_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat5_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = T * _moffat5_fguess[2] * rng.uniform(low=low, high=high)
        guess[7] = T * _moffat5_fguess[3] * rng.uniform(low=low, high=high)
        guess[8] = T * _moffat5_fguess[4] * rng.uniform(low=low, high=high)

        guess[9] = flux * _moffat5_pguess[0] * rng.uniform(low=low, high=high)
        guess[10] = flux * _moffat5_pguess[1] * rng.uniform(low=low, high=high)
        guess[11] = flux * _moffat5_pguess[2] * rng.uniform(low=low, high=high)
        guess[12] = flux * _moffat5_pguess[3] * rng.uniform(low=low, high=high)
        guess[13] = flux * _moffat5_pguess[4] * rng.uniform(low=low, high=high)
        return guess


_moffat2_pguess = np.array([0.5, 0.5])
_moffat2_fguess = np.array([0.48955064, 1.50658978])

_moffat3_pguess = np.array([0.27559669, 0.55817131, 0.166232])
_moffat3_fguess = np.array([0.36123609, 0.8426139, 2.58747785])

_moffat4_pguess = np.array([0.44534, 0.366951, 0.10506, 0.0826497])
_moffat4_fguess = np.array([0.541019, 1.19701, 0.282176, 3.51086])

_moffat5_pguess = np.array([0.45, 0.25, 0.15, 0.1, 0.05])
_moffat5_fguess = np.array([0.541019, 1.19701, 0.282176, 3.51086])

_moffat5_pguess = np.array(
    [0.57874897, 0.32273483, 0.03327272, 0.0341253, 0.03111819]
)
_moffat5_fguess = np.array(
    [0.27831284, 0.9959897, 5.86989779, 5.63590429, 4.17285878]
)


def _fix_guess_TFlux(guess, prior, ntry=4):
    """
    just fix T and flux
    """

    n = guess.shape[0]
    for j in range(n):
        for itry in range(ntry):
            try:
                lnp = prior.get_lnprob_scalar(guess[j, :])

                if lnp <= LOWVAL:
                    dosample = True
                else:
                    dosample = False
            except GMixRangeError:
                dosample = True

            if dosample:
                print_pars(guess[j, :], front="bad guess:", logger=LOGGER)
                if itry < ntry:
                    tguess = prior.sample()
                    guess[j, 4:] = tguess[4:]
                else:
                    # give up and just drawn a sample
                    guess[j, :] = prior.sample()
            else:
                break


def _fix_guess(guess, prior, ntry=4):
    """
    Fix a guess for out-of-bounds values according the the input prior
    Bad guesses are replaced by a sample from the prior
    """

    n = guess.shape[0]
    for j in range(n):
        for itry in range(ntry):
            try:
                lnp = prior.get_lnprob_scalar(guess[j, :])

                if lnp <= LOWVAL:
                    dosample = True
                else:
                    dosample = False
            except GMixRangeError:
                dosample = True

            if dosample:
                print_pars(guess[j, :], front="bad guess:", logger=LOGGER)
                guess[j, :] = prior.sample()
            else:
                break

                
def run_admom(
    obs, guess,
    maxiter=DEFAULT_MAXITER,
    shiftmax=DEFAULT_SHIFTMAX,
    etol=DEFAULT_ETOL,
    Ttol=DEFAULT_TTOL,
    rng=None,
):
    """
    obs: Observation
        ngmix.Observation
    guess: ngmix.GMix or a float
        A guess for the fitter.  Can be a full gaussian mixture or a single
        value for T, in which case the rest of the parameters for the
        gaussian are generated.
    maxiter: integer, optional
        Maximum number of iterations, default 200
    etol: float, optional
        absolute tolerance in e1 or e2 to determine convergence,
        default 1.0e-5
    Ttol: float, optional
        relative tolerance in T <x^2> + <y^2> to determine
        convergence, default 1.0e-3
    shiftmax: float, optional
        Largest allowed shift in the centroid, relative to
        the initial guess.  Default 5.0 (5 pixels if the jacobian
        scale is 1)
    rng: numpy.random.RandomState
        Random state for creating full gaussian guesses based
        on a T guess
    """

    am = AdmomFitter(
        maxiter=maxiter,
        shiftmax=shiftmax,
        etol=etol,
        Ttol=Ttol,
        rng=rng,
    )
    return am.go(obs=obs, guess=guess)


class AdmomResult(dict):
    """
    Represent a fit using adaptive moments, and generate images and mixtures
    for the best fit
    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    result: dict
        the basic fit result, to bad added to this object's keys
    """

    def __init__(self, obs, result):
        self._obs = obs
        self.update(result)

    def get_gmix(self):
        """
        get a gmix representing the best fit, normalized
        """
        if self['flags'] != 0:
            raise RuntimeError('cannot create gmix, fit failed')

        pars = self['pars'].copy()
        pars[5] = 1.0

        e1 = pars[2]/pars[4]
        e2 = pars[3]/pars[4]

        g1, g2 = e1e2_to_g1g2(e1, e2)
        pars[2] = g1
        pars[3] = g2

        return GMixModel(pars, "gauss")

    def make_image(self):
        """
        Get an image of the best fit mixture
        Returns
        -------
        image: array
            Image of the model, including the PSF if a psf was sent
        """
        if self['flags'] != 0:
            raise RuntimeError('cannot create image, fit failed')

        obs = self._obs
        jac = obs.jacobian

        gm = self.get_gmix()
        gm.set_flux(obs.image.sum())

        im = gm.make_image(
            obs.image.shape,
            jacobian=jac,
        )
        return im


class AdmomFitter(object):
    """
    Measure adaptive moments for the input observation
    parameters
    ----------
    maxiter: integer, optional
        Maximum number of iterations, default 200
    etol: float, optional
        absolute tolerance in e1 or e2 to determine convergence,
        default 1.0e-5
    Ttol: float, optional
        relative tolerance in T <x^2> + <y^2> to determine
        convergence, default 1.0e-3
    shiftmax: float, optional
        Largest allowed shift in the centroid, relative to
        the initial guess.  Default 5.0 (5 pixels if the jacobian
        scale is 1)
    rng: numpy.random.RandomState
        Random state for creating full gaussian guesses based
        on a T guess
    """

    def __init__(self,
                 maxiter=DEFAULT_MAXITER,
                 shiftmax=DEFAULT_SHIFTMAX,
                 etol=DEFAULT_ETOL,
                 Ttol=DEFAULT_TTOL,
                 rng=None):

        self._set_conf(maxiter, shiftmax, etol, Ttol)

        self.rng = rng

    def go(self, obs, guess):
        """
        run the adpative moments
        parameters
        ----------
        obs: Observation
            ngmix.Observation
        guess: ngmix.GMix or a float
            A guess for the fitter.  Can be a full gaussian mixture or a single
            value for T, in which case the rest of the parameters for the
            gaussian are generated.
        """

        if not isinstance(obs, Observation):
            raise ValueError("input obs must be an Observation")

        guess_gmix = self._get_guess(obs=obs, guess=guess)

        ares = self._get_am_result()

        wt_gmix = guess_gmix._data
        try:
            admom(
                self.conf,
                wt_gmix,
                obs.pixels,
                ares,
            )
        except GMixRangeError:
            ares['flags'] = 0x8

        result = get_result(ares)

        return AdmomResult(obs=obs, result=result)

    def _get_guess(self, obs, guess):
        if isinstance(guess, GMix):
            guess_gmix = guess
        else:
            Tguess = guess  # noqa
            guess_gmix = self._generate_guess(obs=obs, Tguess=Tguess)
        return guess_gmix

    def _set_conf(self, maxiter, shiftmax, etol, Ttol):  # noqa
        dt = numpy.dtype(_admom_conf_dtype, align=True)
        conf = numpy.zeros(1, dtype=dt)

        conf['maxit'] = maxiter
        conf['shiftmax'] = shiftmax
        conf['etol'] = etol
        conf['Ttol'] = Ttol

        self.conf = conf

    def _get_am_result(self):
        dt = numpy.dtype(_admom_result_dtype, align=True)
        return numpy.zeros(1, dtype=dt)

    def _get_rng(self):
        if self.rng is None:
            self.rng = numpy.random.RandomState()

        return self.rng

    def _generate_guess(self, obs, Tguess):  # noqa

        rng = self._get_rng()

        scale = obs.jacobian.get_scale()
        pars = numpy.zeros(6)
        pars[0:0+2] = rng.uniform(low=-0.5*scale, high=0.5*scale, size=2)
        pars[2:2+2] = rng.uniform(low=-0.3, high=0.3, size=2)
        pars[4] = Tguess*(1.0 + rng.uniform(low=-0.1, high=0.1))
        pars[5] = 1.0

        return GMixModel(pars, "gauss")


def get_result(ares):
    """
    copy the result structure to a dict, and
    calculate a few more things
    """

    if isinstance(ares, numpy.ndarray):
        ares = ares[0]
        names = ares.dtype.names
    else:
        names = list(ares.keys())

    res = {}
    for n in names:
        if n == 'sums':
            res[n] = ares[n].copy()
        elif n == 'sums_cov':
            res[n] = ares[n].reshape((6, 6)).copy()
        else:
            res[n] = ares[n]

    res['flux_mean'] = -9999.0
    res['s2n'] = -9999.0
    res['e'] = numpy.array([-9999.0, -9999.0])
    res['e_err'] = 9999.0

    if res['flags'] == 0:
        flux_sum = res['sums'][5]
        res['flux_mean'] = flux_sum/res['wsum']
        res['pars'][5] = res['flux_mean']

        # now want pars and cov for [cen1,cen2,e1,e2,T,flux]
        sums = res['sums']

        pars = res['pars']
        sums_cov = res['sums_cov']

        res['T'] = pars[4]

        if sums[5] > 0.0:
            # the sums include the weight, so need factor of two to correct
            res['T_err'] = 4*get_ratio_error(
                sums[4],
                sums[5],
                sums_cov[4, 4],
                sums_cov[5, 5],
                sums_cov[4, 5],
            )

        if res['T'] > 0.0:
            res['e'][:] = res['pars'][2:2+2]/res['T']

            sums = res['sums']
            res['e1err'] = 2*get_ratio_error(
                sums[2],
                sums[4],
                sums_cov[2, 2],
                sums_cov[4, 4],
                sums_cov[2, 4],
            )
            res['e2err'] = 2*get_ratio_error(
                sums[3],
                sums[4],
                sums_cov[3, 3],
                sums_cov[4, 4],
                sums_cov[3, 4],
            )

            if (not numpy.isfinite(res['e1err']) or
                    not numpy.isfinite(res['e2err'])):
                res['e1err'] = 9999.0
                res['e2err'] = 9999.0
                res['e_cov'] = diag([9999.0, 9999.0])
            else:
                res['e_cov'] = diag([res['e1err']**2, res['e2err']**2])

        else:
            res['flags'] = 0x8

        fvar_sum = sums_cov[5, 5]

        if fvar_sum > 0.0:

            flux_err = numpy.sqrt(fvar_sum)
            res['s2n'] = flux_sum/flux_err

            # error on each shape component from BJ02 for gaussians
            # assumes round

            res['e_err_r'] = 2.0/res['s2n']
        else:
            res['flags'] = 0x40

    res['flagstr'] = _admom_flagmap[res['flags']]

    return res


_admom_result_dtype = [
    ('flags', 'i4'),
    ('numiter', 'i4'),
    ('nimage', 'i4'),
    ('npix', 'i4'),
    ('wsum', 'f8'),

    ('sums', 'f8', 6),
    ('sums_cov', 'f8', (6, 6)),
    ('pars', 'f8', 6),
    # temporary
    ('F', 'f8', 6),
]

_admom_conf_dtype = [
    ('maxit', 'i4'),
    ('shiftmax', 'f8'),
    ('etol', 'f8'),
    ('Ttol', 'f8'),
]

_admom_flagmap = {
    0: 'ok',
    0x1: 'edge hit',  # not currently used
    0x2: 'center shifted too far',
    0x4: 'flux < 0',
    0x8: 'T < 0',
    0x10: 'determinant near zero',
    0x20: 'maxit reached',
    0x40: 'zero var',
}

        
#from metacal_fitter import MetacalFitter
#from base_fitter import *
from util import *
from procflags import *

import logging
import numpy as np
import esutil as eu

NGMIX_V2 = True




def metacal_bootstrap(
    obs,
    runner,
    psf_runner=None,
    ignore_failed_psf=True,
    rng=None,
    **metacal_kws
):
    """
    Make metacal sheared images and run a fitter/measurment, possibly
    bootstrapping the fit based on information inferred from the data or the
    psf model
    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    runner: ngmix Runner
        Must have go(obs=obs) method
    psf_runner: ngmix PSFRunner, optional
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    rng: numpy.random.RandomState
        Random state for generating noise fields.  Not needed if metacal if
        using the noise field in the observations
    **metacal_kws:  keywords
        Keywords to send to get_all_metacal
    Returns
    -------
    resdict, obsdict
        resdict is keyed by the metacal types (e.g. '1p') and holds results
        for each
        obsdict is keyed by the metacal types and holds the metacal observations
    Side effects
    ------------
    the obs.psf.meta['result'] and the obs.psf.gmix may be set if a psf runner
    is sent and the internal fitter has a get_gmix method.  gmix are only set
    for successful fits
    """

    obsdict = get_all_metacal(obs=obs, rng=rng, **metacal_kws)

    resdict = {}

    for key, tobs in obsdict.items():
        resdict[key] = bootstrap(
            obs=tobs, runner=runner, psf_runner=psf_runner,
            ignore_failed_psf=ignore_failed_psf,
        )
        # resdict[key] = runner.get_result()

    return resdict, obsdict

class MetacalBootstrapper(object):
    """
    Make metacal sheared images and run a fitter/measurment, possibly
    bootstrapping the fit based on information inferred from the data or the
    psf model
    Parameters
    ----------
    runner: fit runner for object
        Must have go(obs=obs) method
    psf_runner: fit runner for psfs
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    rng: numpy.random.RandomState
        Random state for generating noise fields.  Not needed if metacal if
        using the noise field in the observations
    **metacal_kws:  keywords
        Keywords to send to get_all_metacal
    """
    def __init__(self, runner, psf_runner, ignore_failed_psf=True,
                 rng=None,
                 **metacal_kws):
        self.runner = runner
        self.psf_runner = psf_runner
        self.ignore_failed_psf = ignore_failed_psf
        self.metacal_kws = metacal_kws
        self.rng = rng

    def go(self, obs):
        """
        Run the runners on the input observation(s)
        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        """
        return metacal_bootstrap(
            obs=obs,
            runner=self.runner,
            psf_runner=self.psf_runner,
            ignore_failed_psf=self.ignore_failed_psf,
            rng=self.rng,
            **self.metacal_kws
        )

    @property
    def fitter(self):
        """
        get a reference to the fitter
        """
        return self.runner.fitter

    
from ngmix.gaussmom import GaussMom
    
def _fit_one_psf(obs, pconf):
    Tguess = 4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss = ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner = ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
        )

    elif 'em' in pconf['model']:
        ngauss = ngmix.bootstrap.get_em_ngauss(pconf['model'])
        runner = ngmix.bootstrap.EMRunner(
            obs,
            Tguess,
            ngauss,
            pconf['em_pars'],
        )

    else:
        runner = ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
        )

    runner.go(ntry=pconf['ntry'])

    psf_fitter = runner.fitter
    res = psf_fitter.get_result()
    obs.update_meta_data({'fitter': psf_fitter})

    if res['flags'] == 0:
        gmix = psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))


import logging

import ngmix
from ngmix.gexceptions import BootPSFFailure
from ngmix.joint_prior import PriorSimpleSep

try:
    from ngmix.joint_prior import PriorBDFSep, PriorBDSep
    NO_BD_MODELS = False
except ImportError:
    NO_BD_MODELS = True

#logger = logging.getLogger(__name__)


class FitterBase(dict):
    def __init__(self, conf, nband, rng):
        self.nband = nband
        self.rng = rng
        self.update(conf)
        self._setup()

    def go(self, mbobs_list):
        """abstract method to do measurements"""
        raise NotImplementedError("implement go()")

    def _get_prior(self, conf):
        if 'priors' not in conf:
            return None

        ppars = conf['priors']
        if ppars.get('prior_from_mof', False):
            return None

        # g
        gp = ppars['g']
        assert gp['type'] == "ba"
        g_prior = self._get_prior_generic(gp)

        if 'T' in ppars:
            size_prior = self._get_prior_generic(ppars['T'])
        elif 'hlr' in ppars:
            size_prior = self._get_prior_generic(ppars['hlr'])
        else:
            raise ValueError('need T or hlr in priors')

        flux_prior = self._get_prior_generic(ppars['flux'])

        # center
        cp = ppars['cen']
        assert cp['type'] == 'normal2d'
        cen_prior = self._get_prior_generic(cp)

        if 'bd' in conf['model']:
            assert 'fracdev' in ppars, (
                "set fracdev prior for bdf and bd models")
            assert not NO_BD_MODELS, (
                "Using BD models requires ngmix with the proper priors."
                " Try updating!")

        if conf['model'] == 'bd':
            assert 'logTratio' in ppars, "set logTratio prior for bd model"
            fp = ppars['fracdev']
            logTratiop = ppars['logTratio']

            fracdev_prior = self._get_prior_generic(fp)
            logTratio_prior = self._get_prior_generic(logTratiop)

            prior = PriorBDSep(
                cen_prior,
                g_prior,
                size_prior,
                logTratio_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        elif conf['model'] == 'bdf':
            fp = ppars['fracdev']

            fracdev_prior = self._get_prior_generic(fp)

            prior = PriorBDFSep(
                cen_prior,
                g_prior,
                size_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        else:
            prior = PriorSimpleSep(
                cen_prior,
                g_prior,
                size_prior,
                [flux_prior]*self.nband,
            )

        return prior

    def _get_prior_generic(self, ppars):
        """
        get a prior object using the input specification
        """
        ptype = ppars['type']
        bounds = ppars.get('bounds', None)

        if ptype == "flat":
            assert bounds is None, 'bounds not supported for flat'
            prior = ngmix.priors.FlatPrior(*ppars['pars'], rng=self.rng)

        elif ptype == "bounds":
            prior = ngmix.priors.LMBounds(*ppars['pars'], rng=self.rng)

        elif ptype == 'two-sided-erf':
            assert bounds is None, 'bounds not supported for erf'
            prior = ngmix.priors.TwoSidedErf(*ppars['pars'], rng=self.rng)

        elif ptype == 'sinh':
            assert bounds is None, 'bounds not supported for Sinh'
            prior = ngmix.priors.Sinh(
                ppars['mean'], ppars['scale'], rng=self.rng)

        elif ptype == 'normal':
            prior = ngmix.priors.Normal(
                ppars['mean'],
                ppars['sigma'],
                bounds=bounds,
                rng=self.rng,
            )

        elif ptype == 'truncated-normal':
            assert 'do not use truncated normal'
            prior = ngmix.priors.TruncatedGaussian(
                mean=ppars['mean'],
                sigma=ppars['sigma'],
                minval=ppars['minval'],
                maxval=ppars['maxval'],
                rng=self.rng,
            )

        elif ptype == 'log-normal':
            assert bounds is None, 'bounds not yet supported for LogNormal'
            if 'shift' in ppars:
                shift = ppars['shift']
            else:
                shift = None
            prior = ngmix.priors.LogNormal(
                ppars['mean'],
                ppars['sigma'],
                shift=shift,
                rng=self.rng,
            )

        elif ptype == 'normal2d':
            assert bounds is None, 'bounds not yet supported for Normal2D'
            prior = ngmix.priors.CenPrior(
                0.0,
                0.0,
                ppars['sigma'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype == 'ba':
            assert bounds is None, 'bounds not supported for BA'
            prior = ngmix.priors.GPriorBA(ppars['sigma'], rng=self.rng)

        else:
            raise ValueError("bad prior type: '%s'" % ptype)

        return prior


            

logger = logging.getLogger(__name__)

METACAL_TYPES = ['noshear', '1p', '1m', '2p', '2m']


class MetacalFitter(FitterBase):
    """Run metacal on all a list of observations.
    Parameters
    ----------
    conf : dict
        A configuration dictionary.
    nband : int
        The number of bands.
    rng : np.random.RandomState
        An RNG instance.
    Methods
    -------
    go(mbobs_list_input)
    """
    def __init__(self, conf, nband, rng):
        super().__init__(conf, nband, rng)

    def _setup(self):
        self.metacal_prior = self._get_prior(self['metacal'])
        assert self.metacal_prior is not None
        self['metacal']['symmetrize_weight'] = self['metacal'].get(
            'symmetrize_weight', False)

        # be safe friends!
        if 'types' in self['metacal']:
            assert self['metacal']['types'] == METACAL_TYPES

    @property
    def result(self):
        """Get the result data"""
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        if self._result is not None:
            return self._result.copy()
        else:
            return None

    def go(self, mbobs_list):
        """Run metcal on a list of MultiBandObsLists.
        Parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple MultiBandObsList it will
            be converted to a list
        """
        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        mbobs_list_mcal = mbobs_list
        self.mbobs_list_mcal = mbobs_list_mcal

        if self['metacal']['symmetrize_weight']:
            self._symmetrize_weights(mbobs_list_mcal)

        self._result = self._do_all_metacal(mbobs_list_mcal)

    def _symmetrize_weights(self, mbobs_list):
        def _symmetrize_weight(wt):
            assert wt.shape[0] == wt.shape[1]
            wt_rot = np.rot90(wt)
            wzero = wt_rot == 0.0
            if np.any(wzero):
                wt[wzero] = 0.0

        for mbobs in mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    _symmetrize_weight(obs.weight)

    def _do_all_metacal(self, mbobs_list):
        """run metacal on all objects
        NOTE: failed mbobs will have no entries in the final list
        """

        assert len(mbobs_list[0]) == self.nband

        datalist = []
        for i, _mbobs in enumerate(mbobs_list):
            passed_flags, mbobs = self._check_flags(_mbobs)
            if passed_flags:
                try:
                    if NGMIX_V2:
                  
                        res = self._do_one_metacal_ngmixv2(mbobs)
                    else:
      
                        boot = self._do_one_metacal_ngmixv1(mbobs)
                        if isinstance(boot, dict):
                            res = boot
                        else:
                            res = boot.get_metacal_result()
                except (BootPSFFailure, BootGalFailure) as err:
                    logger.debug(str(err))
                    res = {'mcal_flags': 1}

                if res['mcal_flags'] != 0:
                    logger.debug("metacal fit failed")
                else:
                    # make sure we send an array
                    fit_data = self._get_metacal_output(res, self.nband, mbobs)
                    self._print_result(fit_data)
                    datalist.append(fit_data)

        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output

    def _do_one_metacal_ngmixv2(self, mbobs):
        gm = GaussMom(1.2).go(mbobs[0][0])
        if gm['flags'] == 0:
            flux_guess = gm['flux']
            Tguess = gm['T']
        else:
            gm = GaussMom(1.2).go(mbobs[0][0].psf)
            if gm['flags'] == 0:
                Tguess = 2 * gm['T']
            else:
                Tguess = 2
            flux_guess = np.sum(mbobs[0][0].image)

        guesser = TFluxAndPriorGuesser(
            rng=self.rng, T=Tguess, flux=flux_guess, prior=self.metacal_prior,
        )
        psf_guesser = SimplePSFGuesser(rng=self.rng, guess_from_moms=True)

        fitter = Fitter(
            model=self['metacal']['model'],
            fit_pars=self['metacal']['max_pars']['pars']['lm_pars'],
            prior=self.metacal_prior
        )
        psf_fitter = Fitter(
            model=self['metacal']['psf']['model'],
            fit_pars=self['metacal']['psf']['lm_pars'],
        )

        psf_runner = PSFRunner(
            fitter=psf_fitter,
            guesser=psf_guesser,
            ntry=self['metacal']['psf']['ntry'],
        )
        runner = Runner(
            fitter=fitter,
            guesser=guesser,
            ntry=self['metacal']['max_pars']['ntry'],
        )

        boot = MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=self.rng, **self['metacal']['metacal_pars'],
        )
    
        resdict, _ = boot.go(mbobs[0][0])
        flags = 0
        for key in resdict:
            flags |= resdict[key]['flags']
        resdict['mcal_flags'] = flags

        psf_T = 0.0
        psf_g = np.zeros(2)
        wsum = 0.0
        for obslist in mbobs:
            for obs in obslist:
                msk = obs.weight > 0
                if np.any(msk) and 'result' in obs.psf.meta:
                    wgt = np.median(obs.weight[msk])
                    psf_T += wgt * obs.psf.meta['result']['T']
                    psf_g += wgt * obs.psf.meta['result']['e']
                    wsum += wgt
        if wsum > 0:
            psf_T /= wsum
            psf_g /= wsum
            resdict['noshear']['Tpsf'] = psf_T
            resdict['noshear']['gpsf'] = psf_g
        else:
            resdict['mcal_flags'] |= procflags.NO_DATA
            resdict['noshear']['Tpsf'] = -9999.0
            resdict['noshear']['gpsf'] = np.array([-9999.0, -9999.0])

        return resdict

    def _do_one_metacal_ngmixv1(self, mbobs):
        conf = self['metacal']
        psf_pars = conf['psf']
        max_conf = conf['max_pars']

        tpsf_obs = mbobs[0][0].psf
        if not tpsf_obs.has_gmix():
            _fit_one_psf(tpsf_obs, psf_pars)

        psf_Tguess = tpsf_obs.gmix.get_T()

        boot = self._get_bootstrapper(mbobs)
        if 'lm_pars' in psf_pars:
            psf_fit_pars = psf_pars['lm_pars']
        else:
            psf_fit_pars = None

        prior = self.metacal_prior
        guesser = None

        boot.fit_metacal(
            psf_pars['model'],
            conf['model'],
            max_conf['pars'],
            psf_Tguess,
            psf_fit_pars=psf_fit_pars,
            psf_ntry=psf_pars['ntry'],
            prior=prior,
            guesser=guesser,
            ntry=max_conf['ntry'],
            metacal_pars=conf['metacal_pars'],
        )
        return boot

    def _check_flags(self, mbobs):
        flags = self['metacal'].get('bmask_flags', None)
        passed_flags = True
        _mbobs = None

        if flags is not None:
            _mbobs = MultiBandObsList()
            _mbobs.update_meta_data(mbobs.meta)
            for obslist in mbobs:
                _obslist = ObsList()
                _obslist.update_meta_data(obslist.meta)
                
                for obs in [obslist]:
                    msk = (obs.bmask & flags) != 0
                    if np.any(msk):
                        logger.info("   EDGE HIT")
                    else:
                        _obslist.append(obs)
                        passed_flags = True

                _mbobs.append(_obslist)

            # all bands have to have at least one obs
            for ol in _mbobs:
                if len(ol) == 0:
                    passed_flags = False

        return passed_flags, _mbobs

    def _print_result(self, data):
        logger.debug(
            "    mcal s2n: %g Trat: %g",
            data['mcal_s2n_noshear'][0],
            data['mcal_T_ratio_noshear'][0])

    def _get_metacal_dtype(self, npars, nband):
        dt = [
            ('x', 'f8'),
            ('y', 'f8'),
            ('mcal_flags', 'i8'),
        ]
        for mtype in METACAL_TYPES:
            n = Namer(front='mcal', back=mtype)
            if mtype == 'noshear':
                dt += [
                    (n('psf_g'), 'f8', 2),
                    (n('psf_T'), 'f8'),
                ]

            dt += [
                (n('nfev'), 'i4'),
                (n('s2n'), 'f8'),
                (n('s2n_r'), 'f8'),
                (n('pars'), 'f8', npars),
                (n('pars_cov'), 'f8', (npars, npars)),
                (n('g'), 'f8', 2),
                (n('g_cov'), 'f8', (2, 2)),
                (n('T'), 'f8'),
                (n('T_err'), 'f8'),
                (n('T_ratio'), 'f8'),
                (n('flux'), 'f8', (nband,)),
                (n('flux_cov'), 'f8', (nband, nband)),
                (n('flux_err'), 'f8', (nband,)),
            ]

        return dt

    def _get_metacal_output(self, allres, nband, mbobs):
        # assume one epoch and line up in all
        # bands
        # FIXME? IDK why this was here
        # assert len(mbobs[0]) == 1, 'one epoch only'

        # needed for the code below
        assert METACAL_TYPES[0] == 'noshear'

        if 'T_r' in allres['noshear'] and 'T' not in allres['noshear']:
            do_round = True
        else:
            do_round = False

        npars = len(allres['noshear']['pars'])
        dt = self._get_metacal_dtype(npars, nband)
        data = np.zeros(1, dtype=dt)

        data0 = data[0]
        #data0['y'] = mbobs[0][0].meta['orig_row']
        #data0['x'] = mbobs[0][0].meta['orig_col']
        data0['mcal_flags'] = 0

        for mtype in METACAL_TYPES:
            n = Namer(front='mcal', back=mtype)

            res = allres[mtype]

            if mtype == 'noshear':
                data0[n('psf_g')] = res['gpsf']
                if do_round:
                    data0[n('psf_T')] = res['psf_T_r']
                else:
                    data0[n('psf_T')] = res['Tpsf']

            for name in res:
                if do_round and name == 'T_r':
                    nn = n('T')
                else:
                    nn = n(name)
                if nn in data.dtype.names:
                    data0[nn] = res[name]

            # this relies on noshear coming first in the metacal
            # types
            data0[n('T_ratio')] = data0[n('T')]/data0['mcal_psf_T_noshear']

        return data

    def _get_bootstrapper(self, mbobs):
        return MaxMetacalBootstrapper(
            mbobs,
            verbose=False,
        )

    



   

import logging
import os

import numpy as np
import joblib
import esutil as eu
import fitsio
from ngmix import ObsList, MultiBandObsList
from ngmix.gexceptions import GMixRangeError

from ngmix.medsreaders import MultiBandNGMixMEDS, NGMixMEDS

#from .metacal import MetacalFitter
#from .ngmix_compat import NGMIX_V2
#from eastlake.step import Step
#from eastlake.utils import safe_mkdir

logger = logging.getLogger(__name__)

# always and forever
MAGZP_REF = 30.0

CONFIG = {
    'metacal': {
        # check for an edge hit
        'bmask_flags': 2**30,

        'metacal_pars': {
            'psf': 'fitgauss',
            'types': ['noshear', '1p', '1m', '2p', '2m'],
        },

        'model': 'gauss',

        'max_pars': {
            'ntry': 2,
            'pars': {
                'method': 'lm',
                'lm_pars': {
                    'maxfev': 2000,
                    'xtol': 5.0e-5,
                    'ftol': 5.0e-5,
                }
            }
        },

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.263
            },

            'g': {
                'type': 'ba',
                'sigma': 0.2
            },

            'T': {
                'type': 'two-sided-erf',
                'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
            },

            'flux': {
                'type': 'two-sided-erf',
                'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
            }
        },

        'psf': {
            'model': 'gauss',
            'ntry': 2,
            'lm_pars': {
                'maxfev': 2000,
                'ftol': 1.0e-5,
                'xtol': 1.0e-5
            }
        }
    },
}



def _run_metacal(meds_files, seed):
    """Run metacal on a tile.
    Parameters
    ----------
    meds_files : list of str
        A list of the meds files to run metacal on.
    seed : int
        The seed for the global RNG.
    """
    with NGMixMEDS(meds_files[0]) as m:
        cat = m.get_cat()
    logger.info(' meds files %s', meds_files)

    n_cpus = joblib.externals.loky.cpu_count()
    n_chunks = max(n_cpus, 60)
    n_obj_per_chunk = int(cat.size / n_chunks)
    if n_obj_per_chunk * n_chunks < cat.size:
        n_obj_per_chunk += 1
    assert n_obj_per_chunk * n_chunks >= cat.size
    logger.info(
        ' running metacal for %d objects in %d chunks', cat.size, n_chunks)

    seeds = np.random.RandomState(seed=seed).randint(1, 2**30, size=n_chunks)

    jobs = []
    for chunk in range(n_chunks):
        start = chunk * n_obj_per_chunk
        end = min(start + n_obj_per_chunk, cat.size)
        jobs.append(joblib.delayed(_run_mcal_one_chunk)(
            meds_files, start, end, seeds[chunk]))

    with joblib.Parallel(
        n_jobs=n_cpus, backend='multiprocessing',
        verbose=100, max_nbytes=None
    ) as p:
        outputs = p(jobs)

    assert not all([o is None for o in outputs]), (
        "All metacal fits failed!")

    output = eu.numpy_util.combine_arrlist(
        [o for o in outputs if o is not None])
    logger.info(' %d of %d metacal fits worked!', output.size, cat.size)

    return output


def _run_mcal_one_chunk(meds_files, start, end, seed):
    """Run metcal for `meds_files` only for objects from `start` to `end`.
    Note that `start` and `end` follow normal python indexing conventions so
    that the list of indices processed is `list(range(start, end))`.
    Parameters
    ----------
    meds_files : list of str
        A list of paths to the MEDS files.
    start : int
        The starting index of objects in the file on which to run metacal.
    end : int
        One plus the last index to process.
    seed : int
        The seed for the RNG.
    Returns
    -------
    output : np.ndarray
        The metacal outputs.
    """
    rng = np.random.RandomState(seed=seed)

    # seed the global RNG to try to make things reproducible
    np.random.seed(seed=rng.randint(low=1, high=2**30))

    output = None
    mfiles = []
    data = []
    try:
        # get the MEDS interface
        for m in meds_files:
            mfiles.append(NGMixMEDS(m))
        mbmeds = MultiBandNGMixMEDS(mfiles)
        cat = mfiles[0].get_cat()

        for ind in range(start, end):
            o = mbmeds.get_mbobs(ind)
            o = _strip_coadd(o)
            o = _strip_zero_flux(o)
            if not NGMIX_V2:
                # ngmix v1 worked in surface brightness, not flux
                o = _apply_pixel_scale(o)

            skip_me = False
            for ol in o:
                if len(ol) == 0:
                    logger.debug(' not all bands have images - skipping!')
                    skip_me = True
            if skip_me:
                continue

            o.meta['id'] = ind
            o[0].meta['Tsky'] = 1
            o[0].meta['magzp_ref'] = MAGZP_REF
            o[0][0].meta['orig_col'] = cat['orig_col'][ind, 0]
            o[0][0].meta['orig_row'] = cat['orig_row'][ind, 0]

            nband = len(o)
            mcal = MetacalFitter(CONFIG, nband, rng)

            try:
                mcal.go([o])
                res = mcal.result
            except GMixRangeError as e:
                logger.debug(" metacal error: %s", str(e))
                res = None

            if res is not None:
                data.append(res)

        if len(data) > 0:
            output = eu.numpy_util.combine_arrlist(data)
    finally:
        for m in mfiles:
            m.close()

    return output


def _strip_coadd(mbobs):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        for i in range(1, len(ol)):
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs


def _strip_zero_flux(mbobs):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        for i in range(len(ol)):
            if np.sum(ol[i].image) > 0:
                _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs


def _apply_pixel_scale(mbobs):
    for ol in mbobs:
        for o in ol:
            scale = o.jacobian.get_scale()
            scale2 = scale * scale
            scale4 = scale2 * scale2
            o.image = o.image / scale2
            o.weight = o.weight * scale4
    return mbobs
        
    
    
class ObsList(list, MetadataMixin):
    """
    Hold a list of Observation objects
    This class provides a bit of type safety and ease of type checking.
    parameters
    ----------
    meta: dict or None
        Any metadata keep in the `meta` attribute. Optional.
    """

    def __init__(self, meta=None):
        super(ObsList, self).__init__()

        self.set_meta(meta)

    def append(self, obs):
        """
        Add a new observation
        over-riding this for type safety
        parameters
        ----------
        obs: ngmix.Observation
            An observation. An AssertionError will be raised if `obs` is not
            an `ngmix.Observation`.
        """
        mess = "obs should be of type Observation, got %s" % type(obs)
        assert isinstance(obs, Observation), mess
        super(ObsList, self).append(obs)

    def get_s2n(self):
        """
        get the the simple s/n estimator
        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
        returns
        -------
        s2n: float
            The s/n of the images. Will retun -9999 if the s/n cannot be computed.
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/np.sqrt(Vsum)
        else:
            s2n = -9999.0
        return s2n

    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator
        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
        returns
        -------
        -------
        Isum: float
            The value sum(I).
        Vsum: float
            The value sum(1/w)
        Npix: int
            The number of non-zero-weight pixels.
        """

        Isum = 0.0
        Vsum = 0.0
        Npix = 0

        for obs in self:
            tIsum, tVsum, tNpix = obs.get_s2n_sums()
            Isum += tIsum
            Vsum += tVsum
            Npix += tNpix

        return Isum, Vsum, Npix

    def __setitem__(self, index, obs):
        """
        over-riding this for type safety
        """
        assert isinstance(obs, Observation), \
            "obs should be of type Observation"
        super(ObsList, self).__setitem__(index, obs)


class MultiBandObsList(list, MetadataMixin):
    """
    Hold a list of lists of ObsList objects, each representing a filter
    band
    This class provides a bit of type safety and ease of type checking
    parameters
    ----------
    meta: dict or None
        Any metadata keep in the `meta` attribute. Optional.
    """

    def __init__(self, meta=None):
        super(MultiBandObsList, self).__init__()

        self.set_meta(meta)

    def append(self, obs_list):
        """
        Add a new ObsList
        over-riding this for type safety
        parameters
        ----------
        obs_list: ngmix.ObsList
            An ObsList. An AssertionError will be raised if `obs_list` is not
            an `ngmix.ObsList`.
        """
        assert isinstance(obs_list, ObsList),\
            'obs_list should be of type ObsList'
        super(MultiBandObsList, self).append(obs_list)

    def get_s2n(self):
        """
        get the the simple s/n estimator
        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
        returns
        -------
        s2n: float
            The s/n of the images. Will retun -9999 if the s/n cannot be computed.
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/np.sqrt(Vsum)
        else:
            s2n = -9999.0
        return s2n

    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator
        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
        returns
        -------
        Isum: float
            The value sum(I).
        Vsum: float
            The value sum(1/w)
        Npix: int
            The number of non-zero-weight pixels.
        """

        Isum = 0.0
        Vsum = 0.0
        Npix = 0

        for obslist in self:
            tIsum, tVsum, tNpix = obslist.get_s2n_sums()
            Isum += tIsum
            Vsum += tVsum
            Npix += tNpix

        return Isum, Vsum, Npix

    def __setitem__(self, index, obs_list):
        """
        over-riding this for type safety
        """
        assert isinstance(obs_list, ObsList),\
            'obs_list should be of type ObsList'
        super(MultiBandObsList, self).__setitem__(index, obs_list)

        
__all__ = [
    'DEFAULT_STEP',
    'METACAL_TYPES', 'METACAL_MINIMAL_TYPES',
]


# need all these types for psf='dilate'
METACAL_TYPES = [
    'noshear',
    '1p', '1m', '2p', '2m',
]

# these are the types needed when the new psf is round
METACAL_MINIMAL_TYPES = [
    'noshear',
    '1p', '1m', '2p', '2m',
]

DEFAULT_STEP = 0.01
  
        
def get_all_metacal(
    obs,
    psf='gauss',
    step=DEFAULT_STEP,
    fixnoise=True,
    rng=None,
    use_noise_image=False,
    types=None,
):
    """
    Get all combinations of metacal images in a dict
    parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The values in the dict correspond to these
    psf: string or galsim object, optional
        PSF to use for metacal.  Default 'gauss'.  Note 'fitgauss'
        will usually produce a smaller psf, but it can fail.
            'gauss': reconvolve gaussian that is larger than
                the original and round.
            'fitgauss': fit a gaussian to the PSF and make
                use round, dilated version for reconvolution
            galsim object: any arbitrary galsim object
                Use the exact input object for the reconvolution kernel; this
                psf gets convolved by thye pixel
            'dilate': dilate the origial psf
                just dilate the original psf; the resulting psf is not round,
                so you need to calculate the _psf terms and make an explicit
                correction
    step: float, optional
        The shear step value to use for metacal.  Default 0.01
    fixnoise: bool, optional
        If set to True, add a compensating noise field to cancel the effect of
        the sheared, correlated noise component.  Default True
    rng: np.random.RandomState
        A random number generator; this is required if fixnoise is True and
        use_noise_image is False.  It is also required when psf= is sent, in
        order to add a small amount of noise to the rendered image of the
        psf.
    use_noise_image: bool, optional
        If set to True, use the .noise attribute of the observation
        for fixing the noise when fixnoise=True.
    types: list, optional
        If psf='gauss' or 'fitgauss', then the default set is the minimal
        set ['noshear','1p','1m','2p','2m']
        Otherwise, the default is the full possible set listed in
        ['noshear','1p','1m','2p','2m',
         '1p_psf','1m_psf','2p_psf','2m_psf']
    returns
    -------
    A dictionary with all the relevant metacaled images
        dict keys:
            1p -> ( shear, 0)
            1m -> (-shear, 0)
            2p -> ( 0, shear)
            2m -> ( 0, -shear)
        simular for 1p_psf etc.
    """

    if fixnoise:
        odict = _get_all_metacal_fixnoise(
            obs, step=step, rng=rng,
            use_noise_image=use_noise_image,
            psf=psf,
            types=types,
        )
    else:
        logger.debug("    not doing fixnoise")
        odict = _get_all_metacal(
            obs, step=step, rng=rng,
            psf=psf,
            types=types,
        )

    return odict


def _get_all_metacal(
    obs,
    step=DEFAULT_STEP,
    rng=None,
    psf=None,
    types=None,
):
    """
    internal routine
    get all metacal
    """
    if isinstance(obs, Observation):

        if psf == 'dilate':
            m = MetacalDilatePSF(obs)
        else:

            if psf == 'gauss':
                m = MetacalGaussPSF(obs=obs, rng=rng)
            elif psf == 'fitgauss':
                m = MetacalFitGaussPSF(obs=obs, rng=rng)
            else:
                m = MetacalAnalyticPSF(obs=obs, psf=psf, rng=rng)

        odict = m.get_all(step=step, types=types)

    elif isinstance(obs, MultiBandObsList):
        odict = _make_metacal_mb_obs_list_dict(
            mb_obs_list=obs, step=step, rng=rng,
            psf=psf,
            types=types,
        )
    elif isinstance(obs, ObsList):
        odict = _make_metacal_obs_list_dict(
            obs, step, rng=rng,
            psf=psf,
            types=types,
        )
    else:
        raise ValueError("obs must be Observation, ObsList, "
                         "or MultiBandObsList")

    return odict

def _get_all_metacal_fixnoise(
    obs,
    step=DEFAULT_STEP,
    rng=None,
    use_noise_image=False,
    psf=None,
    types=None,
):
    """
    internal routine
    Add a sheared noise field to cancel the correlated noise
    """

    # Using None for the model means we get just noise
    if use_noise_image:
        noise_obs = _replace_image_with_noise(obs)
        logger.debug("    Doing fixnoise with input noise image")
    else:
        noise_obs = simulate_obs(gmix=None, obs=obs, rng=rng)

    # rotate by 90
    _rotate_obs_image_square(noise_obs, k=1)

    obsdict = _get_all_metacal(
        obs, step=step, rng=rng,
        psf=psf,
        types=types,
    )
    noise_obsdict = _get_all_metacal(
        noise_obs, step=step, rng=rng,
        psf=psf,
        types=types,
    )

    for type in obsdict:

        imbobs = obsdict[type]
        nmbobs = noise_obsdict[type]

        # rotate back, which is 3 more rotations
        _rotate_obs_image_square(nmbobs, k=3)

        if isinstance(imbobs, Observation):
            _doadd_single_obs(imbobs, nmbobs)

        elif isinstance(imbobs, ObsList):
            for iobs in range(len(imbobs)):

                obs = imbobs[iobs]
                nobs = nmbobs[iobs]

                _doadd_single_obs(obs, nobs)

        elif isinstance(imbobs, MultiBandObsList):
            for imb in range(len(imbobs)):
                iolist = imbobs[imb]
                nolist = nmbobs[imb]

                for iobs in range(len(iolist)):

                    obs = iolist[iobs]
                    nobs = nolist[iobs]

                    _doadd_single_obs(obs, nobs)

    return obsdict

def simulate_obs(
    gmix, obs,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):
    """
    Simulate the observation(s) using the input gaussian mixture
    parameters
    ----------
    gmix: GMix or subclass
        The gaussian mixture or None
    obs: observation(s)
        One of Observation, ObsList, or MultiBandObsList
    convolve_psf: bool, optional
        If True, convolve by the PSF.  Default True.
    add_noise: bool, optional
        If True, add noise according to the weight map.  Default True.
        The noise image is monkey-patched in as obs.noise_image
    use_raw_weight: bool, optional
        If True, look for a .weight_raw attribute for generating
        the noise.  Often one is using modified weight map to simplify
        masking neighbors, but may want to use the raw map for
        adding noise.  Default True
    add_all: bool, optional
        If True, add noise to zero-weight pixels as well.  For max like methods
        this makes no difference, but if the image is being run through an FFT
        it might be important. Default is True.
    """

    if isinstance(obs, MultiBandObsList):
        return _simulate_mbobs(
            gmix_list=gmix, mbobs=obs,
            add_noise=add_noise,
            rng=rng,
            add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
            convolve_psf=convolve_psf,
        )

    else:

        if gmix is not None and not isinstance(gmix, GMix):
            raise ValueError("input gmix must be a gaussian mixture")

        elif isinstance(obs, ObsList):
            return _simulate_obslist(
                gmix, obs,
                add_noise=add_noise,
                rng=rng,
                add_all=add_all,
                noise_factor=noise_factor,
                use_raw_weight=use_raw_weight,
                convolve_psf=convolve_psf,
            )

        elif isinstance(obs, Observation):
            return _simulate_obs(
                gmix, obs,
                add_noise=add_noise,
                rng=rng,
                add_all=add_all,
                noise_factor=noise_factor,
                use_raw_weight=use_raw_weight,
                convolve_psf=convolve_psf,
            )

        else:
            raise ValueError(
                "obs should be an Observation, " "ObsList, or MultiBandObsList"
            )


def _simulate_mbobs(
    gmix_list, mbobs,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):

    if gmix_list is not None:
        if not isinstance(gmix_list, list):
            raise ValueError(
                "for simulating MultiBandObsLists, the "
                "input must be a list of gaussian mixtures"
            )

        if not isinstance(gmix_list[0], GMix):
            raise ValueError("input must be gaussian mixtures")

        if not len(gmix_list) == len(mbobs):

            mess = "len(mbobs)==%d but len(gmix_list)==%d"
            mess = mess % (len(mbobs), len(gmix_list))
            raise ValueError(mess)

    new_mbobs = MultiBandObsList()
    nband = len(mbobs)
    for i in range(nband):
        if gmix_list is None:
            gmix = None
        else:
            gmix = gmix_list[i]

        ol = mbobs[i]
        new_obslist = _simulate_obslist(
            gmix=gmix, obslist=ol,
            add_noise=add_noise,
            rng=rng,
            add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
            convolve_psf=convolve_psf,
        )
        new_mbobs.append(new_obslist)

    return new_mbobs


def _simulate_obslist(
    gmix, obslist,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):
    new_obslist = ObsList()
    for o in obslist:
        newobs = simulate_obs(
            gmix=gmix, obs=o,
            add_noise=add_noise,
            rng=rng,
            add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
            convolve_psf=convolve_psf,
        )
        new_obslist.append(newobs)

    return new_obslist


def _simulate_obs(
    gmix, obs,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):

    sim_image = _get_simulated_image(gmix, obs, convolve_psf=convolve_psf)

    if add_noise:
        sim_image, noise_image = _get_noisy_image(
            obs, sim_image, rng=rng, add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
        )
    else:
        noise_image = None

    if not obs.has_psf():
        psf = None
    else:
        psf = deepcopy(obs.psf)

    weight = obs.weight.copy()

    if noise_factor is not None:
        LOGGER.debug(
            "Modding weight with noise factor: %s" % noise_factor
        )
        weight *= 1.0 / noise_factor ** 2

    new_obs = Observation(
        sim_image, weight=weight, jacobian=obs.jacobian, psf=psf
    )

    new_obs.noise_image = noise_image
    return new_obs


def _get_simulated_image(gmix, obs, convolve_psf=True):
    if gmix is None:
        return zeros(obs.image.shape)

    if convolve_psf:
        psf_gmix = _get_psf_gmix(obs)

        gm = gmix.convolve(psf_gmix)
    else:
        gm = gmix

    sim_image = gm.make_image(obs.image.shape, jacobian=obs.jacobian)

    return sim_image


def _get_noisy_image(obs, sim_image, rng, add_all=True, noise_factor=None,
                     use_raw_weight=True):
    """
    create a noise image from the weight map
    """

    # often we are using a modified weight map for fitting,
    # to simplify masking of neighbors.  The user can request
    # to use an attribute called `weight_raw` instead, which
    # would have the unmodified weight map, good for adding the
    # correct noise

    if hasattr(obs, "weight_raw") and use_raw_weight:
        weight = obs.weight_raw
    else:
        weight = obs.weight

    noise_image = get_noise_image(
        weight=weight, rng=rng, add_all=add_all, noise_factor=noise_factor,
    )
    return sim_image + noise_image, noise_image


BIGNOISE = 1.0e15


def get_noise_image(weight, rng, add_all=True, noise_factor=None):
    """
    get a noise image based on the input weight map
    If add_all, we set weight==0 pixels with the median noise.  This should not
    be a problem for algorithms that use the weight map
    """

    if rng is None:
        raise ValueError('you must send an rng to get_noise_image')

    noise_image = rng.normal(loc=0.0, scale=1.0, size=weight.shape,)

    err = zeros(weight.shape)
    w = where(weight > 0)
    if w[0].size > 0:
        err[w] = sqrt(1.0 / weight[w])

        if add_all and (w[0].size != weight.size):
            # there were some zero weight pixels, and we
            # want to add noise there anyway
            median_err = numpy.median(err[w])

            wzero = where(weight <= 0)
            err[wzero] = median_err

        if noise_factor is not None:
            LOGGER.debug("Adding noise factor: %s" % noise_factor)
            err *= noise_factor

    else:
        LOGGER.debug("All weight is zero!  Setting noise to %s" % BIGNOISE)
        err[:, :] = BIGNOISE

    noise_image *= err
    return noise_image


def _get_psf_gmix(obs):
    if not obs.has_psf():
        raise RuntimeError(
            "You requested to convolve by the psf, "
            "but the observation has no psf observation set"
        )

    psf = obs.get_psf()
    if not psf.has_gmix():
        raise RuntimeError(
            "You requested to convolve by the psf, "
            "but the observation has no psf gmix set"
        )

    return psf.gmix
from numpy import where, sqrt, zeros
from copy import deepcopy

def _rotate_obs_image_square(obs, k=1):
    """
    rotate the image.  internal routine just for fixnoise with rotnoise=True
    """

    if isinstance(obs, Observation):
        obs.set_image(np.rot90(obs.image, k=k))
    elif isinstance(obs, ObsList):
        for tobs in obs:
            _rotate_obs_image_square(tobs, k=k)
    elif isinstance(obs, MultiBandObsList):
        for obslist in obs:
            _rotate_obs_image_square(obslist, k=k)


def _doadd_single_obs(obs, nobs):
    obs.image_orig = obs.image.copy()
    obs.weight_orig = obs.weight.copy()

    # the weight and image can be modified in the context, and update_pixels is
    # automatically called upon exit

    with obs.writeable():
        obs.image += nobs.image

        wpos = np.where(
            (obs.weight != 0.0) &
            (nobs.weight != 0.0)
        )
        if wpos[0].size > 0:
            tvar = obs.weight*0
            # add the variances
            tvar[wpos] = (
                1.0/obs.weight[wpos] +
                1.0/nobs.weight[wpos]
            )
            obs.weight[wpos] = 1.0/tvar[wpos]


def _replace_image_with_noise(obs):
    """
    copy the observation and copy the .noise parameter
    into the image position
    """

    noise_obs = copy.deepcopy(obs)

    if isinstance(noise_obs, Observation):
        noise_obs.image = noise_obs.noise
    elif isinstance(noise_obs, ObsList):
        for nobs in noise_obs:
            nobs.image = nobs.noise
    else:
        for obslist in noise_obs:
            for nobs in obslist:
                nobs.image = nobs.noise

    return noise_obs


class MetacalDilatePSF(object):
    """
    Create manipulated images for use in metacalibration
    Parameters
    ----------
    obs: ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image
    examples
    --------
    mc = MetacalDilatePSF(obs)
    # observations used to calculate R
    sh1m=ngmix.Shape(-0.01,  0.00 )
    sh1p=ngmix.Shape( 0.01,  0.00 )
    sh2m=ngmix.Shape( 0.00, -0.01 )
    sh2p=ngmix.Shape( 0.00,  0.01 )
    R_obs1m = mc.get_obs_galshear(sh1m)
    R_obs1p = mc.get_obs_galshear(sh1p)
    R_obs2m = mc.get_obs_galshear(sh2m)
    R_obs2p = mc.get_obs_galshear(sh2p)
    # you can also get an unsheared, just convolved obs
    R_obs1m, R_obs1m_unsheared = mc.get_obs_galshear(sh1p, get_unsheared=True)
    # observations used to calculate Rpsf
    Rpsf_obs1m = mc.get_obs_psfshear(sh1m)
    Rpsf_obs1p = mc.get_obs_psfshear(sh1p)
    Rpsf_obs2m = mc.get_obs_psfshear(sh2m)
    Rpsf_obs2p = mc.get_obs_psfshear(sh2p)
    """

    def __init__(self, obs):

        self.obs = obs

        if not obs.has_psf():
            raise ValueError("observation must have a psf observation set")

        self._set_pixel()
        self._set_interp()
        self._set_data()
        self._psf_cache = {}

    def get_all(self, step=DEFAULT_STEP, types=None):
        """
        Get metacal images in a dict for the requested image types
        parameters
        ----------
        step: float
            The shear step value to use for metacal. Default 0.01
        types: list
            Types to get.  Default is the full possible set listed in
            METACAL_TYPES = ['noshear','1p','1m','2p','2m',
                             '1p_psf','1m_psf','2p_psf','2m_psf']
            If you are not using a round PSF, you should also request the
            sheared psf terms to make psf leakage corrections
            ['1p_psf','1m_psf','2p_psf','2m_psf'].  You can get this
            full set in METACAL_TYPES
        returns
        -------
        A dictionary with all the relevant metacaled images, e.g.
            with dict keys:
                noshear -> (0, 0)
                1p -> ( shear, 0)
                1m -> (-shear, 0)
                2p -> ( 0,  shear)
                2m -> ( 0, -shear)
        """

        if types is None:
            types = copy.deepcopy(METACAL_TYPES)
        else:
            for t in types:
                assert t in METACAL_TYPES, 'bad metacal type: %s' % t

        # we add 1p here if we want noshear since we get both of those
        # at once below

        if 'noshear' in types and '1p' not in types:
            types.append('1p')

        shdict = {}

        # galshear keys
        shdict['1m'] = Shape(-step, 0.0)
        shdict['1p'] = Shape(+step, 0.0)

        shdict['2m'] = Shape(0.0, -step)
        shdict['2p'] = Shape(0.0, +step)

        # psfshear keys
        keys = list(shdict.keys())
        for key in keys:
            pkey = '%s_psf' % key
            shdict[pkey] = shdict[key].copy()

        odict = {}

        for type in types:
            if type == 'noshear':
                # we get noshear with 1p
                continue

            sh = shdict[type]

            if 'psf' in type:
                obs = self.get_obs_psfshear(sh)
            else:
                if type == '1p':
                    # add in noshear from this one
                    obs, obs_noshear = self.get_obs_galshear(
                        sh,
                        get_unsheared=True
                    )
                    odict['noshear'] = obs_noshear
                else:
                    obs = self.get_obs_galshear(sh)

            odict[type] = obs

        return odict

    def get_obs_galshear(self, shear, get_unsheared=False):
        """
        This is the case where we shear the image, for calculating R
        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        get_unsheared: bool
            Get an observation only convolved by the target psf, not
            sheared
        """

        type = 'gal_shear'

        newpsf_image, newpsf_obj = self.get_target_psf(shear, type)

        sheared_image = self.get_target_image(newpsf_obj, shear=shear)

        newobs = self._make_obs(sheared_image, newpsf_image)

        # this is the pixel-convolved psf object, used to draw the
        # psf image
        newobs.psf.galsim_obj = newpsf_obj

        if get_unsheared:
            unsheared_image = self.get_target_image(newpsf_obj, shear=None)

            uobs = self._make_obs(unsheared_image, newpsf_image)
            uobs.psf.galsim_obj = newpsf_obj

            return newobs, uobs
        else:
            return newobs

    def get_obs_psfshear(self, shear):
        """
        This is the case where we shear the psf image, for calculating Rpsf
        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        """
        newpsf_image, newpsf_obj = self.get_target_psf(shear, 'psf_shear')
        conv_image = self.get_target_image(newpsf_obj, shear=None)

        newobs = self._make_obs(conv_image, newpsf_image)
        return newobs

    def get_target_psf(self, shear, type):
        """
        get image and galsim object for the dilated, possibly sheared, psf
        parameters
        ----------
        shear: ngmix.Shape
            The applied shear
        type: string
            Type of psf target.  For type='gal_shear', the psf is just dilated
            to deal with noise amplification.  For type='psf_shear' the psf is
            also sheared for calculating Rpsf
        returns
        -------
        image, galsim object
        """

        _check_shape(shear)

        if type == 'psf_shear':
            doshear = True
        else:
            doshear = False

        key = self._get_psf_key(shear, doshear)
        if key not in self._psf_cache:
            psf_grown = self._get_dilated_psf(shear, doshear=doshear)

            # this should carry over the wcs
            psf_grown_image = self.psf_image.copy()

            try:
                psf_grown.drawImage(
                    image=psf_grown_image,
                    method='no_pixel',  # pixel is already in psf
                )
            except RuntimeError as err:
                # argh, galsim uses generic exceptions
                raise GMixRangeError("galsim error: '%s'" % str(err))

            self._psf_cache[key] = (psf_grown_image, psf_grown)

        psf_grown_image, psf_grown = self._psf_cache[key]
        return psf_grown_image.copy(), psf_grown

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm
        If doshear, also shear it
        """
        import galsim

        psf_grown_nopix = self._do_dilate(self.psf_int_nopix, shear, doshear=doshear)

        if doshear:
            psf_grown_nopix = psf_grown_nopix.shear(g1=shear.g1, g2=shear.g2)

        psf_grown = galsim.Convolve(psf_grown_nopix, self.pixel)
        return psf_grown

    def _do_dilate(self, psf, shear, doshear=False):
        key = self._get_psf_key(shear, doshear)
        if key not in self._psf_cache:
            self._psf_cache[key] = _do_dilate(psf, shear)

        return self._psf_cache[key]

    def _get_psf_key(self, shear, doshear):
        g = np.sqrt(shear.g1**2 + shear.g2**2)
        return '%s-%s' % (doshear, g)

    def get_target_image(self, psf_obj, shear=None):
        """
        get the target image, convolved with the specified psf
        and possibly sheared
        parameters
        ----------
        psf_obj: A galsim object
            psf object by which to convolve.  An interpolated image,
            or surface brightness profile
        shear: ngmix.Shape, optional
            The shear to apply
        returns
        -------
        galsim image object
        """

        imconv = self._get_target_gal_obj(psf_obj, shear=shear)

        ny, nx = self.image.array.shape

        try:
            newim = imconv.drawImage(
                nx=nx,
                ny=ny,
                wcs=self.image.wcs,
                dtype=np.float64,
            )
        except RuntimeError as err:
            # argh, galsim uses generic exceptions
            raise GMixRangeError("galsim error: '%s'" % str(err))

        return newim

    def _get_target_gal_obj(self, psf_obj, shear=None):
        import galsim

        if shear is not None:
            shim_nopsf = self.get_sheared_image_nopsf(shear)
        else:
            shim_nopsf = self.image_int_nopsf

        imconv = galsim.Convolve([shim_nopsf, psf_obj])

        return imconv

    def get_sheared_image_nopsf(self, shear):
        """
        get the image sheared by the reqested amount, pre-psf and pre-pixel
        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        returns
        -------
        galsim image object
        """
        _check_shape(shear)
        # this is the interpolated, devonvolved image
        sheared_image = self.image_int_nopsf.shear(g1=shear.g1, g2=shear.g2)
        return sheared_image

    def _set_data(self):
        """
        create galsim objects based on the input observation
        """
        import galsim

        obs = self.obs

        # these would share data with the original numpy arrays, make copies
        # to be sure they don't get modified
        #
        self.image = galsim.Image(obs.image.copy(),
                                  wcs=self.get_wcs())

        self.psf_image = galsim.Image(obs.psf.image.copy(),
                                      wcs=self.get_psf_wcs())

        # interpolated psf image
        psf_int = galsim.InterpolatedImage(self.psf_image,
                                           x_interpolant=self.interp)

        # this can be used to deconvolve the psf from the galaxy image
        psf_int_inv = galsim.Deconvolve(psf_int)

        self.image_int = galsim.InterpolatedImage(self.image,
                                                  x_interpolant=self.interp)

        # deconvolved galaxy image, psf+pixel removed
        self.image_int_nopsf = galsim.Convolve(self.image_int,
                                               psf_int_inv)

        # interpolated psf deconvolved from pixel.  This is what
        # we dilate, shear, etc and reconvolve the image by
        self.psf_int_nopix = galsim.Convolve([psf_int, self.pixel_inv])

    def get_wcs(self):
        """
        get a galsim wcs from the input jacobian
        """
        return self.obs.jacobian.get_galsim_wcs()

    def get_psf_wcs(self):
        """
        get a galsim wcs from the input jacobian
        """
        return self.obs.psf.jacobian.get_galsim_wcs()

    def _set_pixel(self):
        """
        set the pixel based on the pixel scale, for convolutions
        Thanks to M. Jarvis for the suggestion to use toWorld
        to get the proper pixel
        """
        import galsim

        wcs = self.get_wcs()
        self.pixel = wcs.toWorld(galsim.Pixel(scale=1))
        self.pixel_inv = galsim.Deconvolve(self.pixel)

    def _set_interp(self):
        """
        set the laczos interpolation configuration
        """
        self.interp = 'lanczos15'

    def _make_psf_obs(self, psf_im):

        obs = self.obs
        psf_obs = Observation(psf_im.array,
                              weight=obs.psf.weight.copy(),
                              jacobian=obs.psf.jacobian)
        return psf_obs

    def _make_obs(self, im, psf_im):
        """
        Make new Observation objects for the image and psf.
        Copy out the weight maps and jacobians from the original
        Observation.
        parameters
        ----------
        im: Galsim Image
        psf_im: Galsim Image
        returns
        -------
        A new Observation
        """

        obs = self.obs

        psf_obs = self._make_psf_obs(psf_im)
        psf_obs.meta.update(obs.psf.meta)

        meta = {}
        meta.update(obs.meta)
        newobs = Observation(
            im.array,
            jacobian=obs.jacobian,
            weight=obs.weight.copy(),
            psf=psf_obs,
            meta=meta,
        )

        if obs.has_bmask():
            newobs.bmask = obs.bmask

        return newobs


class MetacalGaussPSF(MetacalDilatePSF):
    """
    Create manipulated images for use in metacalibration.  The reconvolution
    kernel is a gaussian generated based on the input psf
    Parameters
    ----------
    obs: ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image
    rng: numpy.random.RandomState
        Random number generator for adding a small amount of noise to the
        gaussian psf image
    examples
    --------
    mc = MetacalGaussPFF(obs)
    # observations used to calculate R
    sh1m=ngmix.Shape(-0.01,  0.00 )
    sh1p=ngmix.Shape( 0.01,  0.00 )
    sh2m=ngmix.Shape( 0.00, -0.01 )
    sh2p=ngmix.Shape( 0.00,  0.01 )
    R_obs1m = mc.get_obs_galshear(sh1m)
    R_obs1p = mc.get_obs_galshear(sh1p)
    R_obs2m = mc.get_obs_galshear(sh2m)
    R_obs2p = mc.get_obs_galshear(sh2p)
    # you can also get an unsheared, just convolved obs
    R_obs1m, R_obs1m_unsheared = mc.get_obs_galshear(sh1p, get_unsheared=True)
    """

    def __init__(self, obs, rng):

        super().__init__(obs=obs)
        if rng is None:
            raise ValueError('send an rng to MetacalGaussPSF')

        self.rng = rng
        self._setup_psf_noise()

    def get_all(self, step=DEFAULT_STEP, types=None):
        """
        Get metacal images in a dict for the requested image types
        parameters
        ----------
        step: float
            The shear step value to use for metacal. Default 0.01
        types: list
            Types to get.  Default is the full possible set listed in
            METACAL_MINIMAL_TYPES = ['noshear','1p','1m','2p','2m']
        returns
        -------
        A dictionary with all the relevant metacaled images, e.g.
            with dict keys:
                noshear -> (0, 0)
                1p -> ( shear, 0)
                1m -> (-shear, 0)
                2p -> ( 0,  shear)
                2m -> ( 0, -shear)
        """

        if types is None:
            types = copy.deepcopy(METACAL_MINIMAL_TYPES)
        else:
            for t in types:
                assert t in METACAL_MINIMAL_TYPES, 'bad metacal type: %s' % t

        return super().get_all(step=step, types=types)

    def _setup_psf_noise(self):
        pim = self.obs.psf.image
        self.psf_flux = pim.sum()

        self.psf_noise = pim.max()/50000.0

        self.psf_noise_image = self.rng.normal(
            size=pim.shape,
            scale=self.psf_noise,
        )
        self.psf_weight = self.psf_noise_image*0 + 1.0/self.psf_noise**2

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm
        """
        import galsim

        assert doshear is False, 'no shearing gauss psf'

        gauss_psf = _get_gauss_target_psf(
            self.psf_int_nopix, flux=self.psf_flux,
        )
        psf_grown_nopix = _do_dilate(gauss_psf, shear)
        psf_grown = galsim.Convolve(psf_grown_nopix, self.pixel)
        return psf_grown

    def _make_psf_obs(self, gsim):

        psf_im = gsim.array.copy()
        psf_im += self.psf_noise_image

        obs = self.obs

        cen = (np.array(psf_im.shape) - 1.0)/2.0

        jacobian = obs.jacobian.copy()
        
  
        
        jacobian.set_cen(row=cen[0], col=cen[1])

        psf_obs = Observation(
            psf_im,
            weight=self.psf_weight,
            jacobian=jacobian,
        )
        return psf_obs


class MetacalFitGaussPSF(MetacalGaussPSF):
    """
    Create manipulated images for use in metacalibration.  The reconvolution
    kernel is a gaussian generated based a fit to the input psf
    Parameters
    ----------
    obs: ngmix.Observation
        Observation on which to run metacal
    rng: numpy.random.RandomState
        Random number generator for adding a small amount of noise to the
        gaussian psf image
    examples
    --------
    rng = np.random.RandomState(seed)
    mc = MetacalFitGaussPSF(obs, rng)
    # observations used to calculate R
    sh1m=ngmix.Shape(-0.01,  0.00 )
    sh1p=ngmix.Shape( 0.01,  0.00 )
    sh2m=ngmix.Shape( 0.00, -0.01 )
    sh2p=ngmix.Shape( 0.00,  0.01 )
    R_obs1m = mc.get_obs_galshear(sh1m)
    R_obs1p = mc.get_obs_galshear(sh1p)
    R_obs2m = mc.get_obs_galshear(sh2m)
    R_obs2p = mc.get_obs_galshear(sh2p)
    # you can also get an unsheared, just convolved obs
    R_obs1m, R_obs1m_unsheared = mc.get_obs_galshear(sh1p, get_unsheared=True)
    """
    def __init__(self, obs, rng):
        super().__init__(obs=obs, rng=rng)
        self._do_psf_fit()

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm
        """

        assert doshear is False, 'no shearing fitgauss psf'

        psf_grown = _do_dilate(self.gauss_psf, shear)

        # we don't convolve by the pixel, its already in there
        return psf_grown

    def _do_psf_fit(self):
        """
        do the gaussian fit.
        try the following in order
            - adaptive moments
            - maximim likelihood
            - see if there is already a gmix object
        if the above all fail, rase BootPSFFailure
        """
        import galsim



        psfobs = self.obs.psf

        ntry = 4
        guesser = GMixPSFGuesser(rng=self.rng, ngauss=1)

        # try adaptive moments first
        fitter = AdmomFitter(rng=self.rng)

        res = run_psf_fitter(obs=psfobs, fitter=fitter, guesser=guesser, ntry=ntry)

        if res['flags'] == 0:
            psf_gmix = res.get_gmix()
        else:
            # try maximum likelihood

            lm_pars = {
                'maxfev': 2000,
                'ftol': 1.0e-05,
                'xtol': 1.0e-05,
            }

            fitter = Fitter(model='gauss', fit_pars=lm_pars)
            guesser = SimplePSFGuesser(rng=self.rng)

            res = run_psf_fitter(
                obs=psfobs, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=False,
            )

            if res['flags'] == 0:
                psf_gmix = res.get_gmix()
            else:

                # see if there was already a gmix that we might use instead
                if psfobs.has_gmix() and len(psfobs.gmix) == 1:
                    psf_gmix = psfobs.gmix.copy()
                else:
                    # ok, just raise and exception
                    raise BootPSFFailure('failed to fit psf '
                                         'for MetacalFitGaussPSF')

        e1, e2, T = psf_gmix.get_e1e2T()

        dilation = _get_ellip_dilation(e1, e2, T)
        T_dilated = T*dilation
        sigma = np.sqrt(T_dilated/2.0)

        self.gauss_psf = galsim.Gaussian(
            sigma=sigma,
            flux=self.psf_flux,
        )


class MetacalAnalyticPSF(MetacalGaussPSF):
    """
    Create manipulated images for use in metacalibration.  The reconvolution
    kernel is set to the input galsim object
    Parameters
    ----------
    obs: ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image
    psf: galsim GSObjec
        The psf used for reconvolution
    rng: numpy.random.RandomState
        Random number generator for adding a small amount of noise to the
        gaussian psf image
    examples
    --------
    mc = MetacalGaussPFF(obs)
    # observations used to calculate R
    sh1m=ngmix.Shape(-0.01,  0.00 )
    sh1p=ngmix.Shape( 0.01,  0.00 )
    sh2m=ngmix.Shape( 0.00, -0.01 )
    sh2p=ngmix.Shape( 0.00,  0.01 )
    R_obs1m = mc.get_obs_galshear(sh1m)
    R_obs1p = mc.get_obs_galshear(sh1p)
    R_obs2m = mc.get_obs_galshear(sh2m)
    R_obs2p = mc.get_obs_galshear(sh2p)
    # you can also get an unsheared, just convolved obs
    R_obs1m, R_obs1m_unsheared = mc.get_obs_galshear(sh1p, get_unsheared=True)
    """
    def __init__(self, obs, psf, rng):
        import galsim
        super().__init__(obs=obs, rng=rng)

        assert isinstance(psf, galsim.GSObject)
        self.psf_obj = psf

    def _get_dilated_psf(self, shear, doshear=False):
        """
        For this version we never pixelize the input
        analytic model
        """

        assert doshear is False, 'no shearing analytic psf'

        psf_grown = _do_dilate(self.psf_obj, shear)
        return psf_grown


def _get_ellip_dilation(e1, e2, T):
    """
    when making a new image after shearing, we need to dilate the PSF to hide
    modes that get exposed
    """
    irr, irc, icc = ngmix.moments.e2mom(e1, e2, T)

    mat = np.zeros((2, 2))
    mat[0, 0] = irr
    mat[0, 1] = irc
    mat[1, 0] = irc
    mat[1, 1] = icc

    eigs = np.linalg.eigvals(mat)

    dilation = eigs.max()/(T/2.)
    dilation = np.sqrt(dilation)

    dilation = 1.0 + 2*(dilation-1.0)

    if dilation > 1.1:
        dilation = 1.1

    return dilation


def _do_dilate(obj, shear):
    """
    Dilate the input Galsim image object according to
    the input shear
    dilation = 1.0 + 2.0*|g|
    parameters
    ----------
    obj: Galsim Image or object
        The object to dilate
    shear: ngmix.Shape
        The shape to use for dilation
    """
    g = np.sqrt(shear.g1**2 + shear.g2**2)
    dilation = 1.0 + 2.0*g
    return obj.dilate(dilation)


def _check_shape(shape):
    """
    ensure the input is an instantiation of ngmix.Shape
    """
    if not isinstance(shape, Shape):
        raise TypeError("shape must be of type ngmix.Shape")


def _get_gauss_target_psf(psf, flux):
    """
    taken from galsim/tests/test_metacal.py
    assumes the psf is centered
    """
    import galsim

    dk = psf.stepk/4.0

    small_kval = 1.e-2    # Find the k where the given psf hits this kvalue
    smaller_kval = 3.e-3  # Target PSF will have this kvalue at the same k

    kim = psf.drawKImage(scale=dk)
    karr_r = kim.real.array
    # Find the smallest r where the kval < small_kval
    nk = karr_r.shape[0]
    kx, ky = np.meshgrid(np.arange(-nk/2, nk/2), np.arange(-nk/2, nk/2))
    ksq = (kx**2 + ky**2) * dk**2
    ksq_max = np.min(ksq[karr_r < small_kval * psf.flux])

    # We take our target PSF to be the (round) Gaussian that is even smaller at
    # this ksq
    # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
    sigma_sq = -2. * np.log(smaller_kval) / ksq_max

    return galsim.Gaussian(sigma=np.sqrt(sigma_sq), flux=flux)
from ngmix.gmix import GMix



from numba import njit

from ngmix.gmix.gmix_nb import (
    gmix_set_norms,
    gmix_eval_pixel_fast,
    GMIX_LOW_DETVAL,
)

ADMOM_EDGE = 0x1
ADMOM_SHIFT = 0x2
ADMOM_FAINT = 0x4
ADMOM_SMALL = 0x8
ADMOM_DET = 0x10
ADMOM_MAXIT = 0x20


@njit
def admom(confarray, wt, pixels, resarray):
    """
    run the adaptive moments algorithm
    parameters
    ----------
    conf: admom config struct
        See admom._admom_conf_dtype
    """
    # to simplify notation
    conf = confarray[0]
    res = resarray[0]

    roworig = wt['row'][0]
    colorig = wt['col'][0]

    e1old = e2old = Told = -9999.0
    for i in range(conf['maxit']):

        if wt['det'][0] < GMIX_LOW_DETVAL:
            res['flags'] = ADMOM_DET
            break

        # due to check above, this should not raise an exception
        gmix_set_norms(wt)

        clear_result(res)
        admom_censums(wt, pixels, res)

        if res['sums'][5] <= 0.0:
            res['flags'] = ADMOM_FAINT
            break

        wt['row'][0] = res['sums'][0]/res['sums'][5]
        wt['col'][0] = res['sums'][1]/res['sums'][5]

        if (abs(wt['row'][0]-roworig) > conf['shiftmax']
                or abs(wt['col'][0]-colorig) > conf['shiftmax']):
            res['flags'] = ADMOM_SHIFT
            break

        clear_result(res)
        admom_momsums(wt, pixels, res)

        if res['sums'][5] <= 0.0:
            res['flags'] = ADMOM_FAINT
            break

        # look for convergence
        finv = 1.0/res['sums'][5]
        M1 = res['sums'][2]*finv
        M2 = res['sums'][3]*finv
        T = res['sums'][4]*finv

        Irr = 0.5*(T - M1)
        Icc = 0.5*(T + M1)
        Irc = 0.5*M2

        if T <= 0.0:
            res['flags'] = ADMOM_SMALL
            break

        e1 = (Icc - Irr)/T
        e2 = 2*Irc/T

        if ((abs(e1-e1old) < conf['etol'])
                and (abs(e2-e2old) < conf['etol'])
                and (abs(T/Told-1.) < conf['Ttol'])):

            res['pars'][0] = wt['row'][0]
            res['pars'][1] = wt['col'][0]
            res['pars'][2] = wt['icc'][0] - wt['irr'][0]
            res['pars'][3] = 2.0*wt['irc'][0]
            res['pars'][4] = wt['icc'][0] + wt['irr'][0]
            res['pars'][5] = 1.0

            break

        else:
            # deweight moments and go to the next iteration

            deweight_moments(wt, Irr, Irc, Icc, res)
            if res['flags'] != 0:
                break

            e1old = e1
            e2old = e2
            Told = T

    res['numiter'] = i

    if res['numiter'] == conf['maxit']:
        res['flags'] = ADMOM_MAXIT


@njit
def admom_censums(wt, pixels, res):
    """
    do sums for determining the center
    """

    n_pixels = pixels.size
    for i in range(n_pixels):

        pixel = pixels[i]
        weight = gmix_eval_pixel_fast(wt, pixel)

        wdata = weight*pixel['val']

        res['npix'] += 1
        res['sums'][0] += wdata*pixel['v']
        res['sums'][1] += wdata*pixel['u']
        res['sums'][5] += wdata


@njit
def admom_momsums(wt, pixels, res):
    """
    do sums for calculating the weighted moments
    """

    vcen = wt['row'][0]
    ucen = wt['col'][0]
    F = res['F']

    n_pixels = pixels.size
    for i_pixel in range(n_pixels):

        pixel = pixels[i_pixel]
        weight = gmix_eval_pixel_fast(wt, pixel)

        var = 1.0/(pixel['ierr']*pixel['ierr'])

        vmod = pixel['v']-vcen
        umod = pixel['u']-ucen

        wdata = weight*pixel['val']
        w2 = weight*weight

        F[0] = pixel['v']
        F[1] = pixel['u']
        F[2] = umod*umod - vmod*vmod
        F[3] = 2*vmod*umod
        F[4] = umod*umod + vmod*vmod
        F[5] = 1.0

        res['wsum'] += weight
        res['npix'] += 1

        for i in range(6):
            res['sums'][i] += wdata*F[i]
            for j in range(6):
                res['sums_cov'][i, j] += w2*var*F[i]*F[j]


@njit
def deweight_moments(wt, Irr, Irc, Icc, res):
    """
    deweight a set of weighted moments
    parameters
    ----------
    wt: gaussian mixture
        The weight used to measure the moments
    Irr, Irc, Icc:
        The weighted moments
    res: admom result struct
        the flags field will be set on error
    """
    # measured moments
    detm = Irr*Icc - Irc*Irc
    if detm <= GMIX_LOW_DETVAL:
        res['flags'] = ADMOM_DET
        return

    Wrr = wt['irr'][0]
    Wrc = wt['irc'][0]
    Wcc = wt['icc'][0]
    detw = Wrr*Wcc - Wrc*Wrc
    if detw <= GMIX_LOW_DETVAL:
        res['flags'] = ADMOM_DET
        return

    idetw = 1.0/detw
    idetm = 1.0/detm

    # Nrr etc. are actually of the inverted covariance matrix
    Nrr = Icc*idetm - Wcc*idetw
    Ncc = Irr*idetm - Wrr*idetw
    Nrc = -Irc*idetm + Wrc*idetw
    detn = Nrr*Ncc - Nrc*Nrc

    if detn <= GMIX_LOW_DETVAL:
        res['flags'] = ADMOM_DET
        return

    # now set from the inverted matrix
    idetn = 1./detn
    wt['irr'][0] = Ncc*idetn
    wt['icc'][0] = Nrr*idetn
    wt['irc'][0] = -Nrc*idetn
    wt['det'][0] = (
        wt['irr'][0]*wt['icc'][0] - wt['irc'][0]*wt['irc'][0]
    )


@njit
def clear_result(res):
    """
    clear some fields in the result structure
    """
    res['npix'] = 0
    res['wsum'] = 0.0
    res['sums'][:] = 0.0
    res['sums_cov'][:, :] = 0.0
    res['pars'][:] = -9999.0

    # res['flags']=0
    # res['numiter']=0
    # res['nimage']=0
    # res['F'][:]=0.0
    
    



def format_pars(pars, fmt="%8.3g"):
    """
    get a nice string of the pars with no line breaks
    Parameters
    ----------
    pars: array/sequence
        The parameters to print
    fmt: string
        The format string for each number
    Returns
    --------
    the string
    """
    fmt = " ".join([fmt + " "] * len(pars))
    return fmt % tuple(pars)


def get_ratio_var(a, b, var_a, var_b, cov_ab):
    """
    get (a/b)**2 and variance in mean of (a/b)
    """

    if b == 0:
        raise ValueError("zero in denominator")

    rsq = (a/b)**2

    var = rsq * (var_a/a**2 + var_b/b**2 - 2*cov_ab/(a*b))
    return var


def get_ratio_error(a, b, var_a, var_b, cov_ab):
    """
    get a/b and error on a/b
    """
    from math import sqrt

    var = get_ratio_var(a, b, var_a, var_b, cov_ab)

    if var < 0:
        var = 0
    error = sqrt(var)
    return error
from numpy import diag
from ngmix.shape import e1e2_to_g1g2
from ngmix import GMixModel

ONE_MINUS_EPS = 0.9999999999999999


def shear_reduced(g1, g2, s1, s2):
    """
    addition formula for reduced shear
    parameters
    ----------
    g1,g2: scalar or array
        "reduced shear" shapes
    s1,s2: scalar or array
        "reduced shear" shapes to use as shear
    outputs
    -------
    g1,g2 after shear
    """

    A = 1 + g1 * s1 + g2 * s2
    B = g2 * s1 - g1 * s2
    denom_inv = 1.0 / (A * A + B * B)

    g1o = A * (g1 + s1) + B * (g2 + s2)
    g2o = A * (g2 + s2) - B * (g1 + s1)

    g1o *= denom_inv
    g2o *= denom_inv

    return g1o, g2o


class Shape(object):
    """
    Shape object.  Currently only for reduced shear style shapes
    examples
    --------
    >>> import numpy as np
    >>> from ngmix.shape import Shape
    >>> s = Shape(0.1, 0.2)
    >>> neg_s = -s
    >>> rot_s = s.get_rotated(np.pi/2)
    >>> new_s = s.copy()
    >>> sheared_s = s.get_sheared(-0.05, 0.0)
    parameters
    ----------
    g1,g2: scalar
        "reduced shear" shapes
    """

    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

        # can't call the other jitted methods
        g = numpy.sqrt(g1 * g1 + g2 * g2)
        if g >= 1.0:
            raise GMixRangeError("g out of range: %.16g" % g)
        self.g = g

    def set_g1g2(self, g1, g2):
        """
        Set reduced shear style ellipticity
        parameters
        ----------
        g1,g2: scalar
            "reduced shear" shapes
        """
        self.g1 = g1
        self.g2 = g2

        g = numpy.sqrt(g1 * g1 + g2 * g2)
        if g >= 1.0:
            raise GMixRangeError("g out of range: %.16g" % g)
        self.g = g

    def get_sheared(self, s1, s2=None):
        """
        Get a new shape, sheared by the specified amount.
        parameters
        ----------
        s1: scalar or Shape
            The first component of the shape or a Shape instance.
        s2: scalar
            If s1 is given as a scalar, you must send the second component
            of the shape as s2.
        outputs
        -------
        sheared_shape: Shape
            A new shape sheared by (s1, s2).
        """

        if isinstance(s1, Shape):
            sh = s1
            s1 = sh.g1
            s2 = sh.g2
        else:
            if s2 is None:
                raise ValueError("send s1,s2 or a Shape")

        g1, g2 = shear_reduced(self.g1, self.g2, s1, s2)
        return Shape(g1, g2)

    def __neg__(self):
        """
        get Shape(-g1, -g2)
        """
        return Shape(-self.g1, -self.g2)

    def get_rotated(self, theta_radians):
        """
        Rotate the shape by the input angle.
        parameters
        ----------
        theta_radians: scalar
            The rotation angle in radians.
        outputs
        -------
        rot_shape: Shape
            The rotated shape.
        """
        twotheta = 2.0 * theta_radians

        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)
        g1rot = self.g1 * cos2angle + self.g2 * sin2angle
        g2rot = -self.g1 * sin2angle + self.g2 * cos2angle

        return Shape(g1rot, g2rot)

    def rotate(self, theta_radians):
        """
        In-place rotation of the shape by the input angle
        **deprecated, use get_rotated()**
        parameters
        ----------
        theta_radians: scalar
            The rotation angle in radians.
        """
        twotheta = 2.0 * theta_radians

        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)
        g1rot = self.g1 * cos2angle + self.g2 * sin2angle
        g2rot = -self.g1 * sin2angle + self.g2 * cos2angle

        self.set_g1g2(g1rot, g2rot)

    def copy(self):
        """
        Make a new Shape object with the same ellipticity parameters.
        outputs
        -------
        new_shape: Shape
            A copy of the current Shape instance.
        """
        s = Shape(self.g1, self.g2)
        return s

    def __repr__(self):
        return "(%.16g, %.16g)" % (self.g1, self.g2)


def g1g2_to_e1e2(g1, g2):
    """
    convert reduced shear g1,g2 to standard ellipticity
    parameters e1,e2
    uses eta representation but could also use
        e1 = 2*g1/(1 + g1**2 + g2**2)
        e2 = 2*g2/(1 + g1**2 + g2**2)
    parameters
    ----------
    g1,g2: scalars
        Reduced shear space shapes
    outputs
    -------
    e1,e2: tuple of scalars
        shapes in (ixx-iyy)/(ixx+iyy) style space
    """
    g = numpy.sqrt(g1 * g1 + g2 * g2)

    if isinstance(g1, numpy.ndarray):
        (w,) = numpy.where(g >= 1.0)
        if w.size != 0:
            raise GMixRangeError("some g were out of bounds")

        eta = 2 * numpy.arctanh(g)
        e = numpy.tanh(eta)

        numpy.clip(e, 0.0, ONE_MINUS_EPS, e)

        e1 = numpy.zeros(g.size)
        e2 = numpy.zeros(g.size)
        (w,) = numpy.where(g != 0.0)
        if w.size > 0:
            fac = e[w] / g[w]

            e1[w] = fac * g1[w]
            e2[w] = fac * g2[w]

    else:
        if g >= 1.0:
            raise GMixRangeError("g out of bounds: %s" % g)
        if g == 0.0:
            return (0.0, 0.0)

        eta = 2 * numpy.arctanh(g)
        e = numpy.tanh(eta)
        if e >= 1.0:
            e = ONE_MINUS_EPS

        fac = e / g

        e1 = fac * g1
        e2 = fac * g2

    return e1, e2


def e1e2_to_g1g2(e1, e2):
    """
    convert e1,e2 to reduced shear style ellipticity
    parameters
    ----------
    e1,e2: tuple of scalars
        shapes in (ixx-iyy)/(ixx+iyy) style space
    outputs
    -------
    g1,g2: scalars
        Reduced shear space shapes
    """

    e = numpy.sqrt(e1 * e1 + e2 * e2)
    if isinstance(e1, numpy.ndarray):
        (w,) = numpy.where(e >= 1.0)
        if w.size != 0:
            raise GMixRangeError("some e were out of bounds")

        eta = numpy.arctanh(e)
        g = numpy.tanh(0.5 * eta)

        numpy.clip(g, 0.0, ONE_MINUS_EPS, g)

        g1 = numpy.zeros(g.size)
        g2 = numpy.zeros(g.size)
        (w,) = numpy.where(e != 0.0)
        if w.size > 0:
            fac = g[w] / e[w]

            g1[w] = fac * e1[w]
            g2[w] = fac * e2[w]

    else:
        if e >= 1.0:
            raise GMixRangeError("e out of bounds: %s" % e)
        if e == 0.0:
            g1, g2 = 0.0, 0.0

        else:

            eta = numpy.arctanh(e)
            g = numpy.tanh(0.5 * eta)

            if g >= 1.0:
                # round off?
                g = ONE_MINUS_EPS

            fac = g / e

            g1 = fac * e1
            g2 = fac * e2

    return g1, g2


def g1g2_to_eta1eta2(g1, g2):
    """
    convert reduced shear g1,g2 to eta style ellipticity
    parameters
    ----------
    g1,g2: scalars
        Reduced shear space shapes
    outputs
    -------
    eta1,eta2: tuple of scalars
        eta space shapes
    """

    if isinstance(g1, numpy.ndarray):

        g = numpy.sqrt(g1 * g1 + g2 * g2)
        (w,) = numpy.where(g >= 1.0)
        if w.size != 0:
            raise GMixRangeError("some g were out of bounds")

        eta1 = numpy.zeros(g.size)
        eta2 = eta1.copy()

        (w,) = numpy.where(g > 0.0)
        if w.size > 0:

            eta = 2 * numpy.arctanh(g[w])
            fac = eta / g[w]

            eta1[w] = fac * g1[w]
            eta2[w] = fac * g2[w]

    else:
        g = numpy.sqrt(g1 * g1 + g2 * g2)

        if g >= 1.0:
            raise GMixRangeError("g out of bounds: %s converting to eta" % g)

        if g == 0.0:
            eta1, eta2 = 0.0, 0.0
        else:

            eta = 2 * numpy.arctanh(g)

            fac = eta / g

            eta1 = fac * g1
            eta2 = fac * g2

    return eta1, eta2


def e1e2_to_eta1eta2(e1, e2):
    """
    convert reduced shear e1,e2 to eta style ellipticity
    parameters
    ----------
    e1,e2: scalars
        Reduced shear space shapes
    outputs
    -------
    eta1,eta2: tuple of scalars
        eta space shapes
    """

    if not isinstance(e1, numpy.ndarray):
        e1 = numpy.array(e1, ndmin=1, copy=False)
        e2 = numpy.array(e2, ndmin=1, copy=False)
        is_scalar = True
    else:
        is_scalar = False

    e = numpy.sqrt(e1 * e1 + e2 * e2)
    (w,) = numpy.where(e >= 1.0)
    if w.size != 0:
        raise GMixRangeError("some e were out of bounds")

    eta1 = numpy.zeros(e.size)
    eta2 = eta1.copy()

    (w,) = numpy.where(e > 0.0)
    if w.size > 0:

        eta = numpy.arctanh(e)
        fac = eta[w] / e[w]

        eta1[w] = fac * e1[w]
        eta2[w] = fac * e2[w]

    if is_scalar:
        eta1 = eta1[0]
        eta2 = eta2[0]

    return eta1, eta2


def eta1eta2_to_g1g2(eta1, eta2):
    """
    convert eta style shpaes to reduced shear shapes
    parameters
    ----------
    eta1,eta2: tuple of scalars
        eta space shapes
    outputs
    -------
    g1,g2: scalars
        Reduced shear space shapes
    """

    if not isinstance(eta1, numpy.ndarray):
        eta1 = numpy.array(eta1, ndmin=1, copy=False)
        eta2 = numpy.array(eta2, ndmin=1, copy=False)
        is_scalar = True
    else:
        is_scalar = False

    g1 = numpy.zeros(eta1.size)
    g2 = g1.copy()

    eta = numpy.sqrt(eta1 * eta1 + eta2 * eta2)

    g = numpy.tanh(0.5 * eta)

    (w,) = numpy.where(g >= 1.0)
    if w.size != 0:
        raise GMixRangeError("some g were out of bounds")

    (w,) = numpy.where(eta != 0.0)
    if w.size > 0:
        fac = g[w] / eta[w]

        g1[w] = fac * eta1[w]
        g2[w] = fac * eta2[w]

    if is_scalar:
        g1 = g1[0]
        g2 = g2[0]

    return g1, g2


def dgs_by_dgo_jacob(g1, g2, s1, s2):
    """
    jacobian of the transformation
        |dgs/dgo|_{shear}
    parameters
    ----------
    g1,g2: numbers or arrays
        shape pars for "observed" image
    s1,s2: numbers or arrays
        shape pars for shear, applied negative
    outputs
    -------
    jacobian : number or array
        The jacobian of the transformation.
    """

    ssq = s1 * s1 + s2 * s2
    num = (ssq - 1) ** 2
    denom = (
        1 + 2 * g1 * s1 + 2 * g2 * s2 + g1 ** 2 * ssq + g2 ** 2 * ssq
    ) ** 2

    jacob = num / denom
    return jacob


def get_round_factor(g1, g2):
    """
    factor to convert T to round T under shear
    Use by taking T_round = T * get_round_factor(g1, g2)
    parameters
    ----------
    g1,g2: scalars
        Reduced shear space shapes
    outputs
    -------
    f: scalar
        factor to convert T to round T under shear
    """
    gsq = g1 ** 2 + g2 ** 2
    f = (1 - gsq) / (1 + gsq)
    return f


def rotate_shape(g1, g2, theta):
    """
    rotate the shapes by the input angle
    parameters
    ----------
    g1: scalar or array
        Shape to be rotated
    g2: scalar or array
        Shape to be rotated
    theta: scalar or array
        Angle in radians
    outputs
    -------
    g1,g2 after rotation
    """

    twotheta = 2.0 * theta

    cos2angle = numpy.cos(twotheta)
    sin2angle = numpy.sin(twotheta)
    g1rot = g1 * cos2angle + g2 * sin2angle
    g2rot = -g1 * sin2angle + g2 * cos2angle

    return g1rot, g2rot


BOOT_S2N_LOW = 2 ** 0
BOOT_R2_LOW = 2 ** 1
BOOT_R4_LOW = 2 ** 2
BOOT_TS2N_ROUND_FAIL = 2 ** 3
BOOT_ROUND_CONVOLVE_FAIL = 2 ** 4
BOOT_WEIGHTS_LOW = 2 ** 5

logger = logging.getLogger(__name__)


class Bootstrapper(object):
    """
    bootstrap fits to psf and object
    Parameters
    ----------
    runner: fit runner for object
        Must have go(obs=obs) method
    psf_runner: fit runner for psfs
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    """
    def __init__(self, runner, psf_runner=None, ignore_failed_psf=True):
        self.runner = runner
        self.psf_runner = psf_runner
        self.ignore_failed_psf = ignore_failed_psf

    def go(self, obs):
        """
        Run the runners on the input observation(s)
        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        """
        return bootstrap(
            obs=obs,
            runner=self.runner,
            psf_runner=self.psf_runner,
            ignore_failed_psf=self.ignore_failed_psf,
        )

    @property
    def fitter(self):
        """
        get a reference to the fitter
        """
        return self.runner.fitter


def bootstrap(
    obs,
    runner,
    psf_runner=None,
    ignore_failed_psf=True,
):
    """
    Run a fitter on the input observations, possibly bootstrapping the fit
    based on information inferred from the data or the psf model
    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    runner: ngmix Runner
        Must have go(obs=obs) method
    psf_runner: ngmix PSFRunner, optional
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    Side effects
    ------------
    the obs.psf.meta['result'] and the obs.psf.gmix may be set if a psf runner
    is sent and the internal fitter has a get_gmix method.  gmix are only set
    for successful fits
    """

    if psf_runner is not None:
        psf_runner.go(obs=obs)

        if ignore_failed_psf:
            obs = remove_failed_psf_obs(obs=obs)

    return runner.go(obs=obs)


def remove_failed_psf_obs(obs):
    """
    remove observations from the input that failed
    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    Returns
    --------
    obs: observation(s)
        new observations, same type as input
    """
    if isinstance(obs, MultiBandObsList):
        new_mbobs = MultiBandObsList(meta=obs.meta)
        for tobslist in obs:

            new_obslist = ObsList(meta=tobslist.meta)
            for tobs in tobslist:
                if tobs.psf.meta['result']['flags'] == 0:
                    new_obslist.append(tobs)

            if len(new_obslist) == 0:
                raise BootPSFFailure('no good psf fits')

            new_mbobs.append(new_obslist)

        return new_mbobs

    elif isinstance(obs, ObsList):
        new_obslist = ObsList(meta=obs.meta)
        for tobs in obs:
            if tobs.psf.meta['result']['flags'] == 0:
                new_obslist.append(tobs)

        if len(new_obslist) == 0:
            raise BootPSFFailure('no good psf fits')

        return new_obslist
    elif isinstance(obs, Observation):
        if obs.psf.meta['result']['flags'] != 0:
            raise BootPSFFailure('no good psf fits')
        return obs
    else:
        mess = (
            'got obs input type: "%s", should be '
            'Observation, ObsList, or MulitiBandObsList' % type(obs)
        )
        raise ValueError(mess)

class FitModel(dict):
    """
    A class to represent a fitting model, the result of the fit, as well as
    generate images and mixtures for the best fit model
    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    model: str
        The model to fit
    prior: ngmix prior
        A prior for fitting
    """

    def __init__(self, obs, model, guess, prior=None):
        self.prior = prior
        self.model = gmix.get_model_num(model)
        self.model_name = gmix.get_model_name(self.model)
        self['model'] = self.model_name

        self._set_obs(obs)
        self._set_totpix()
        self._set_npars()

        self._set_n_prior_pars()
        self._set_fdiff_size()
        self._make_pixel_list()

        self._set_bounds()
        self._setup_fit(guess)

    def set_fit_result(self, result):
        """
        Get some fit statistics for the input pars.
        """

        self.update(result)

        if self["flags"] == 0:
            cres = self.calc_lnprob(self['pars'], more=True)
            self.update(cres)

            if self["s2n_denom"] > 0:
                s2n = self["s2n_numer"] / np.sqrt(self["s2n_denom"])
            else:
                s2n = 0.0

            chi2 = self["lnprob"] / (-0.5)
            dof = self["npix"] - self.npars
            chi2per = chi2 / dof

            self["chi2per"] = chi2per
            self["dof"] = dof
            self["s2n_w"] = s2n
            self["s2n"] = s2n

            self._set_g()
            self._set_T()
            self._set_flux()

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the fit parameter set, which
        definition depends on the sub-class
        Parameters
        ----------
        band: int, optional
            Band index, default 0
        """
        pars = self.get_band_pars(pars=self["pars"], band=band)
        return gmix.make_gmix_model(pars, self.model)

    def get_convolved_gmix(self, band=0, obsnum=0):
        """
        get a gaussian mixture at the fit parameters, convolved by the psf if
        fitting a pre-convolved model
        Parameters
        ----------
        band: int, optional
            Band index, default 0
        obsnum: int, optional
            Number of observation for the given band,
            default 0
        """

        gm = self.get_gmix(band)

        obs = self.obs[band][obsnum]
        if obs.has_psf_gmix():
            gm = gm.convolve(obs.psf.gmix)

        return gm

    def make_image(self, band=0, obsnum=0):
        """
        Get an image of the best fit mixture
        Returns
        -------
        image: array
            Image of the model, including the PSF if a psf was sent
        """
        gm = self.get_convolved_gmix(band=band, obsnum=obsnum)
        obs = self.obs[band][obsnum]
        return gm.make_image(
            obs.image.shape,
            jacobian=obs.jacobian,
        )

    def get_band_pars(self, pars, band):
        """
        get pars for the specified band
        Parameters
        ----------
        pars: array-like
            Array-like of parameters
        band: int
            The band as an integer
        Returns
        -------
        parameters just associated with the requested band
        """
        return get_band_pars(model=self.model_name, pars=pars, band=band)

    def calc_lnprob(self, pars, more=False):
        """
        This is all we use for mcmc approaches, but also used generally for the
        "set_fit_stats" method.  For the max likelihood fitter we also have a
        _get_ydiff method
        Parameters
        ----------
        pars: array-like
            Array-like of parameters
        more: bool
            If true, a dict with more information is returned
        Returns
        -------
        the log(probability) unless more=True, in which case a dict
        with keys
            lnprob: log(probability)
            s2n_numer: numerator for S/N
            s2n_denom: denominator for S/N
            npix: number of pixels used
        """

        try:

            # these are the log pars (if working in log space)
            ln_priors = self._get_priors(pars)

            lnprob = 0.0
            s2n_numer = 0.0
            s2n_denom = 0.0
            npix = 0

            self._fill_gmix_all(pars)
            for band in range(self.nband):

                obs_list = self.obs[band]
                gmix_list = self._gmix_all[band]

                for obs, gm in zip(obs_list, gmix_list):

                    res = gm.get_loglike(obs, more=more)

                    if more:
                        lnprob += res["loglike"]
                        s2n_numer += res["s2n_numer"]
                        s2n_denom += res["s2n_denom"]
                        npix += res["npix"]
                    else:
                        lnprob += res

            # total over all bands
            lnprob += ln_priors

        except GMixRangeError:
            lnprob = LOWVAL
            s2n_numer = 0.0
            s2n_denom = BIGVAL
            npix = 0

        if more:
            return {
                "lnprob": lnprob,
                "s2n_numer": s2n_numer,
                "s2n_denom": s2n_denom,
                "npix": npix,
            }
        else:
            return lnprob

    @property
    def bounds(self):
        """
        get bounds used for fitting with leastsqsbound
        """
        return copy.deepcopy(self._bounds)

    def _set_obs(self, obs_in):
        """
        Set the obs attribute based on the input.
        Parmameters
        -----------
        obs: Observation, ObsList, or MultiBandObsList
            The input observations
        """

        self.obs = get_mb_obs(obs_in)

        self.nband = len(self.obs)
        nimage = 0
        for obslist in self.obs:
            for obs in obslist:
                nimage += 1
        self.nimage = nimage

    def _set_totpix(self):
        """
        Set the total number of pixels
        """

        totpix = 0
        for obs_list in self.obs:
            for obs in obs_list:
                totpix += obs.pixels.size

        self.totpix = totpix

    def _set_npars(self):
        """
        Set the number of parameters.  nband should be set in set_lists, called
        before this
        """
        self.npars = gmix.get_model_npars(self.model) + self.nband - 1

    def _make_model(self, band_pars):
        gm0 = gmix.make_gmix_model(band_pars, self.model)
        return gm0

    def _init_gmix_all(self, pars):
        """
        input pars are in linear space
        initialize the list of lists of gaussian mixtures
        """

        if self.obs[0][0].has_psf_gmix():
            self.dopsf = True
        else:
            self.dopsf = False

        gmix_all0 = MultiBandGMixList()
        gmix_all = MultiBandGMixList()

        for band, obs_list in enumerate(self.obs):
            gmix_list0 = GMixList()
            gmix_list = GMixList()

            # pars for this band, in linear space
            band_pars = self.get_band_pars(pars=pars, band=band)

            for obs in obs_list:
                gm0 = self._make_model(band_pars)
                if self.dopsf:
                    psf_gmix = obs.psf.gmix
                    gm = gm0.convolve(psf_gmix)
                else:
                    gm = gm0.copy()

                gmix_list0.append(gm0)
                gmix_list.append(gm)

            gmix_all0.append(gmix_list0)
            gmix_all.append(gmix_list)

        self._gmix_all0 = gmix_all0
        self._gmix_all = gmix_all

    def _convolve_gmix(self, gm, gm0, psf_gmix):
        """
        norms get set
        """
        gmix_convolve_fill(gm._data, gm0._data, psf_gmix._data)

    def _fill_gmix_all(self, pars):
        """
        input pars are in linear space
        Fill the list of lists of gmix objects for the given parameters
        """

        if not self.dopsf:
            self._fill_gmix_all_nopsf(pars)
            return

        for band, obs_list in enumerate(self.obs):
            gmix_list0 = self._gmix_all0[band]
            gmix_list = self._gmix_all[band]

            band_pars = self.get_band_pars(pars=pars, band=band)

            for i, obs in enumerate(obs_list):

                psf_gmix = obs.psf.gmix

                gm0 = gmix_list0[i]
                gm = gmix_list[i]

                gm0._fill(band_pars)
                self._convolve_gmix(gm, gm0, psf_gmix)

    def _fill_gmix_all_nopsf(self, pars):
        """
        Fill the list of lists of gmix objects for the given parameters
        """

        for band, obs_list in enumerate(self.obs):
            gmix_list = self._gmix_all[band]

            band_pars = self.get_band_pars(pars=pars, band=band)

            for i, obs in enumerate(obs_list):

                gm = gmix_list[i]

                gm._fill(band_pars)

    def _get_priors(self, pars):
        """
        get the sum of ln(prob) from the priors or 0.0 if
        no priors were sent
        """
        if self.prior is None:
            return 0.0
        else:
            return self.prior.get_lnprob_scalar(pars)

    def _setup_fit(self, guess):
        """
        setup the mixtures based on the initial guess
        """

        guess = np.array(guess, dtype="f8", copy=False)

        npars = guess.size
        mess = "guess has npars=%d, expected %d" % (npars, self.npars)
        assert npars == self.npars, mess

        try:
            # this can raise GMixRangeError
            self._init_gmix_all(guess)
            self._make_gmix_list()
        except ZeroDivisionError:
            raise GMixRangeError("got zero division")

    def _set_n_prior_pars(self):
        # center1 + center2 + shape + T + fluxes
        if self.prior is None:
            self.n_prior_pars = 0
        else:
            self.n_prior_pars = get_lm_n_prior_pars(
                model=self.model_name, nband=self.nband,
            )

    def _set_fdiff_size(self):
        self.fdiff_size = self.totpix + self.n_prior_pars

    def _set_bounds(self):
        """
        get bounds on parameters
        """
        self._bounds = None
        if self.prior is not None:
            if hasattr(self.prior, "bounds"):
                self._bounds = self.prior.bounds

    def _set_g(self):
        self["g"] = self["pars"][2:2+2].copy()
        self["g_cov"] = self["pars_cov"][2:2+2, 2:2+2].copy()
        self["g_err"] = self["pars_err"][2:2+2].copy()

    def _set_T(self):
        self["T"] = self["pars"][4]
        self["T_err"] = np.sqrt(self["pars_cov"][4, 4])

    def _set_flux(self):
        _set_flux(res=self, nband=self.nband)

    def _make_pixel_list(self):
        """
        lists of references.
        """
        pixels_list = []

        for band in range(self.nband):
            obs_list = self.obs[band]
            for obs in obs_list:
                pixels_list.append(obs._pixels)

        self._pixels_list = pixels_list

    def _make_gmix_list(self):
        """
        lists of references.
        """
        gmix_data_list = []

        for band in range(self.nband):

            gmix_list = self._gmix_all[band]

            for gm in gmix_list:
                gmdata = gm.get_data()
                gmix_data_list.append(gmdata)

        self._gmix_data_list = gmix_data_list

    def calc_fdiff(self, pars):
        """
        vector with (model-data)/error.
        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)

        try:

            # all norms are set after fill
            self._fill_gmix_all(pars)

            start = self._fill_priors(pars=pars, fdiff=fdiff)

            for pixels, gm in zip(self._pixels_list, self._gmix_data_list):
                fill_fdiff(
                    gm, pixels, fdiff, start,
                )

                start += pixels.size

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff

    def _fill_priors(self, pars, fdiff):
        """
        Fill priors at the beginning of the array.
        ret the position after last par
        We require all the lnprobs are < 0, equivalent to
        the peak probability always being 1.0
        I have verified all our priors have this property.
        """

        if self.prior is None:
            nprior = 0
        else:
            nprior = self.prior.fill_fdiff(pars, fdiff)

        return nprior


class CoellipFitModel(FitModel):
    """
    A class to represent a fitting a coelliptical gaussians model, the result
    of the fit, as well as generate images and mixtures for the best fit model
    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    model: str
        The model to fit
    prior: ngmix prior
        A prior for fitting
    """

    def __init__(self, obs, ngauss, guess, prior=None):
        self._ngauss = ngauss
        super().__init__(obs=obs, model='coellip', guess=guess, prior=prior)

    def _set_flux(self):
        """
        this should be doable
        """
        pass

    def _set_n_prior_pars(self):
        assert self.nband == 1, "Coellip can only fit one band"

        if self.prior is None:
            self.n_prior_pars = 0
        else:
            ngauss = self._ngauss
            self.n_prior_pars = 1 + 1 + 1 + ngauss + ngauss

    def _set_npars(self):
        """
        single band, npars determined from ngauss
        """
        self.npars = 4 + 2 * self._ngauss

    def get_band_pars(self, pars, band):
        """
        Get linear pars for the specified band
        """

        return pars.copy()


class PSFFluxFitModel(dict):
    """
    A class to represent fitting a psf flux fitter.  The class can be
    used as a generic template flux fitter as well
    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    do_psf: bool, optional
        If True, use the gaussian mixtures in the psf observation as templates.
        In this mode the code calculates a "psf flux".  Default True.
    normalize_psf: True or False
        if True, then normalize PSF gmix to flux of unity, otherwise use input
        normalization.  Default True
    """

    def __init__(self, obs, do_psf=True, normalize_psf=True):
        self.do_psf = do_psf

        self.normalize_psf = normalize_psf

        self['model'] = 'template'
        self.npars = 1
        self._set_obs(obs)

    def go(self):
        """
        calculate the flux using zero-lag cross-correlation
        """

        flags = 0

        xcorr_sum = 0.0
        msq_sum = 0.0

        chi2 = 0.0

        nobs = len(self.obs)

        flux = PDEF
        flux_err = CDEF

        for ipass in [1, 2]:
            for iobs in range(nobs):
                obs = self.obs[iobs]

                im = obs.image
                wt = obs.weight

                if ipass == 1:
                    model = self._get_model(iobs)
                    xcorr_sum += (model * im * wt).sum()
                    msq_sum += (model * model * wt).sum()
                else:
                    model = self._get_model(iobs, flux=flux)

                    chi2 += self._get_chi2(model, im, wt)

            if ipass == 1:
                if msq_sum == 0:
                    break
                flux = xcorr_sum / msq_sum

        # chi^2 per dof and error checking
        dof = self.get_dof()
        chi2per = 9999.0
        if dof > 0:
            chi2per = chi2 / dof
        else:
            flags |= ZERO_DOF

        # final flux calculation with error checking
        if msq_sum == 0 or self.totpix == 1:
            flags |= DIV_ZERO
        else:

            arg = chi2 / msq_sum / (self.totpix - 1)
            if arg >= 0.0:
                flux_err = np.sqrt(arg)
            else:
                flags |= BAD_VAR

        result = {
            "flags": flags,
            "chi2per": chi2per,
            "dof": dof,
            "flux": flux,
            "flux_err": flux_err,
        }
        self.update(result)

    def _get_chi2(self, model, im, wt):
        """
        get the chi^2 for this image/model
        we can simulate when needed
        """
        chi2 = ((model - im) ** 2 * wt).sum()
        return chi2

    def _get_model(self, iobs, flux=None):
        """
        get the model image
        """
        if self.use_template:
            if flux is not None:
                model = self.template_list[iobs].copy()
                norm = self.norm_list[iobs]
                model *= (norm * flux) / model.sum()
            else:
                model = self.template_list[iobs]

        else:

            if flux is None:
                gm = self.gmix_list[iobs]
            else:
                gm = self.gmix_list[iobs].copy()
                norm = self.norm_list[iobs]
                gm.set_flux(flux * norm)

            obs = self.obs[iobs]
            dims = obs.image.shape
            jac = obs.jacobian
            model = gm.make_image(dims, jacobian=jac)

        return model

    def get_dof(self):
        """
        Effective def based on effective number of pixels
        """
        npix = self.get_effective_npix()
        dof = npix - self.npars
        if dof <= 0:
            dof = 1.0e-6
        return dof

    def _set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList
        """

        if isinstance(obs_in, Observation):
            obs_list = ObsList()
            obs_list.append(obs_in)
        elif isinstance(obs_in, ObsList):
            obs_list = obs_in
        else:
            raise ValueError("obs should be Observation or ObsList")

        tobs = obs_list[0]
        if self.do_psf:
            tobs = tobs.psf

        if not tobs.has_gmix():
            if not hasattr(tobs, "template"):
                raise ValueError("neither gmix or template image are set")

        self.obs = obs_list
        if tobs.has_gmix():
            self._set_gmix_and_norms()
        else:
            self._set_templates_and_norms()

        self._set_totpix()

    def _set_gmix_and_norms(self):
        self.use_template = False
        gmix_list = []
        norm_list = []
        for obs in self.obs:
            # these return copies, ok to modify
            if self.do_psf:
                gmix = obs.get_psf_gmix()
                if self.normalize_psf:
                    gmix.set_flux(1.0)
            else:
                gmix = obs.get_gmix()
                gmix.set_flux(1.0)

            gmix_list.append(gmix)
            norm_list.append(gmix.get_flux())

        self.gmix_list = gmix_list
        self.norm_list = norm_list

    def _set_templates_and_norms(self):
        self.use_template = True

        template_list = []
        norm_list = []
        for obs in self.obs:
            if self.do_psf:
                template = obs.psf.template.copy()
                norm = template.sum()
                if self.normalize_psf:
                    template *= 1.0 / norm
                    norm = 1.0
            else:
                template = obs.template.copy()
                template *= 1.0 / template.sum()
                norm = 1.0

            template_list.append(template)
            norm_list.append(norm)

        self.template_list = template_list
        self.norm_list = norm_list

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix = 0
        for obs in self.obs:
            totpix += obs.pixels.size

        self.totpix = totpix

    def get_effective_npix(self):
        """
        We don't use all pixels, only those with weight > 0
        """
        if not hasattr(self, "eff_npix"):

            npix = 0
            for obs in self.obs:
                wt = obs.weight

                w = np.where(wt > 0)
                npix += w[0].size

            self.eff_npix = npix

        return self.eff_npix


def get_band_pars(model, pars, band):
    """
    extract parameters for given band
    Parameters
    ----------
    model: string
        Model name
    pars: all parameters
        Parameters for all bands
    band: int
        Band number
    Returns
    -------
    the subset of parameters for this band
    """

    num = gmix.get_model_npars(model)
    band_pars = np.zeros(num)

    assert model != 'coellip'

    if model == 'bd':
        band_pars[0:7] = pars[0:7]
        band_pars[7] = pars[7 + band]
    elif model == 'bdf':
        band_pars = np.zeros(num)
        band_pars[0:6] = pars[0:6]
        band_pars[6] = pars[6 + band]
    else:
        band_pars[0:5] = pars[0:5]
        band_pars[5] = pars[5 + band]

    return band_pars


def get_lm_n_prior_pars(model, nband):
    """
    get the number of slots for priors in LM
    Parameters
    ----------
    model: string
        The model being fit
    nband: int
        Number of bands
    prior: joint prior, optional
        If None, the result is always zero
    """

    if model == 'bd':
        # center1 + center2 + shape + T + log10(Td/Te) + fracdev + fluxes
        npp = 1 + 1 + 1 + 1 + 1 + 1 + nband
    elif model == 'bdf':
        # center1 + center2 + shape + T + fracdev + fluxes
        npp = 1 + 1 + 1 + 1 + 1 + nband
    elif model in ['exp', 'dev', 'gauss', 'turb']:
        # simple models
        npp = 1 + 1 + 1 + 1 + 1 + nband
    else:
        raise ValueError('bad model: %s' % model)

    return npp


def _set_flux(res, nband):
    """
    set the flux in the result dict for standard models.
    Does not work for coellip
    Parameters
    ----------
    res: dict
        The result dict.  Must contain 'pars'
    nband: int
        Number of bands
    """
    from numpy import sqrt, diag

    model = res['model']
    assert model != 'coellip'

    if model == 'bd':
        start = 7
    elif model == 'bdf':
        start = 6
    else:
        start = 5

    if nband == 1:
        res["flux"] = res["pars"][start]
        res["flux_err"] = np.sqrt(res["pars_cov"][start, start])
    else:
        res["flux"] = res["pars"][start:]
        res["flux_cov"] = res["pars_cov"][start:, start:]
        res["flux_err"] = sqrt(diag(res["flux_cov"]))
def get_mb_obs(obs_in):
    """
    convert the input to a MultiBandObsList
    parameters
    ----------
    obs_in: ngmix.Observation, ngmix.ObsList, or ngmix.MultiBandObsList
        Input data to convert to a MultiBandObsList.
    returns
    -------
    mbobs: ngmix.MultiBandObsList
        A MultiBandObsList containing the input data.
    """

    if isinstance(obs_in, Observation):
        obs_list = ObsList()
        obs_list.append(obs_in)

        obs = MultiBandObsList()
        obs.append(obs_list)

    elif isinstance(obs_in, ObsList):
        obs = MultiBandObsList()
        obs.append(obs_in)

    elif isinstance(obs_in, MultiBandObsList):
        obs = obs_in

    else:
        raise ValueError(
            'obs should be Observation, ObsList, or MultiBandObsList'
        )

    return obs
class GMixList(list):
    """
    Hold a list of GMix objects
    This class provides a bit of type safety and ease of type checking
    """

    def append(self, gmix):
        """
        Add a new mixture
        over-riding this for type safety
        """
        assert isinstance(gmix, GMix), "gmix should be of type GMix"
        super(GMixList, self).append(gmix)

    def __setitem__(self, index, gmix):
        """
        over-riding this for type safety
        """
        assert isinstance(gmix, GMix), "gmix should be of type GMix"
        super(GMixList, self).__setitem__(index, gmix)


class MultiBandGMixList(list):
    """
    Hold a list of lists of GMixList objects, each representing a filter
    band
    This class provides a bit of type safety and ease of type checking
    """

    def append(self, gmix_list):
        """
        add a new GMixList
        over-riding this for type safety
        """
        assert isinstance(
            gmix_list, GMixList
        ), "gmix_list should be of type GMixList"
        super(MultiBandGMixList, self).append(gmix_list)

    def __setitem__(self, index, gmix_list):
        """
        over-riding this for type safety
        """
        assert isinstance(
            gmix_list, GMixList
        ), "gmix_list should be of type GMixList"
        super(MultiBandGMixList, self).__setitem__(index, gmix_list)

from numpy import take, eye, triu, transpose, dot, finfo
from numpy import empty_like, sqrt, cos, sin, arcsin, asarray
from numpy import atleast_1d, shape, issubdtype, dtype, inexact
from scipy.optimize import _minpack, leastsq
#from ..util import print_pars
#from ..flags import (
#    ZERO_DOF,
#    DIV_ZERO,
#    LM_SINGULAR_MATRIX,
#    LM_NEG_COV_EIG,
#    LM_NEG_COV_DIAG,
#    EIG_NOTFINITE,
#    LM_FUNC_NOTFINITE,
#)
# default values
PDEF = -9.999e9  # parameter defaults
CDEF = 9.999e9  # covariance or error defaults

# for priors etc.
LOWVAL = -np.inf
BIGVAL = 9999.0e47

DEFAULT_LM_PARS = {"maxfev": 4000, "ftol": 1.0e-5, "xtol": 1.0e-5}



def run_leastsq(func, guess, n_prior_pars, **keys):
    """
    run leastsq from scipy.optimize.  Deal with certain types of errors.  Allow
    the user to input boundaries for parameters, dealt with in a robust way
    using the leastsqbound code that wraps leastsq
    Parameters
    ----------
    func:
        the function to minimize
    guess:
        guess at pars
    n_prior_pars:
        number of slots in fdiff for priors
    bounds : list, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction.
    some useful keywords
    maxfev:
        maximum number of function evaluations. e.g. 1000
    epsfcn:
        Step for jacobian estimation (derivatives). 1.0e-6
    ftol:
        Relative error desired in sum of squares, 1.0e06
    xtol:
        Relative error desired in solution. 1.0e-6
    """

    npars = guess.size
    k_space = keys.pop("k_space", False)

    res = {}
    try:
        lm_tup = leastsqbound(func, guess, full_output=1, **keys)

        pars, pcov0, infodict, errmsg, ier = lm_tup

        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        flags = 0
        if ier > 4:
            flags |= 2 ** (ier - 5)
            pars, pcov, perr = _get_def_stuff(npars)
            #LOGGER.debug(errmsg)

        elif pcov0 is None:
            # why on earth is this not in the flags?
            flags |= LM_SINGULAR_MATRIX
            errmsg = "singular covariance"
            #LOGGER.debug(errmsg)
            #print_pars(pars, front="    pars at singular:", logger=LOGGER)
            junk, pcov, perr = _get_def_stuff(npars)
        else:
            # Scale the covariance matrix returned from leastsq; this will
            # recover the covariance of the parameters in the right units.
            fdiff = func(pars)

            # npars: to remove priors

            if k_space:
                dof = (fdiff.size - n_prior_pars) // 2 - npars
            else:
                dof = fdiff.size - n_prior_pars - npars

            if dof == 0:
                junk, pcov, perr = _get_def_stuff(npars)
                flags |= ZERO_DOF
            else:
                s_sq = (fdiff[n_prior_pars:] ** 2).sum() / dof
                pcov = pcov0 * s_sq

                cflags = _test_cov(pcov)
                if cflags != 0:
                    flags |= cflags
                    errmsg = "bad covariance matrix"
                    #LOGGER.debug(errmsg)
                    junk1, junk2, perr = _get_def_stuff(npars)
                else:
                    # only if we reach here did everything go well
                    perr = sqrt(diag(pcov))

        res["flags"] = flags
        res["nfev"] = infodict["nfev"]
        res["ier"] = ier
        res["errmsg"] = errmsg

        res["pars"] = pars
        res["pars_err"] = perr
        res["pars_cov0"] = pcov0
        res["pars_cov"] = pcov

    except ValueError as e:
        serr = str(e)
        if "NaNs" in serr or "infs" in serr:
            pars, pcov, perr = _get_def_stuff(npars)

            res["pars"] = pars
            res["pars_cov0"] = pcov
            res["pars_cov"] = pcov
            res["nfev"] = -1
            res["flags"] = LM_FUNC_NOTFINITE
            res["errmsg"] = "not finite"
            LOGGER.debug("not finite")
        else:
            raise e

    except ZeroDivisionError:
        pars, pcov, perr = _get_def_stuff(npars)

        res["pars"] = pars
        res["pars_cov0"] = pcov
        res["pars_cov"] = pcov
        res["nfev"] = -1

        res["flags"] = DIV_ZERO
        res["errmsg"] = "zero division"
        LOGGER.debug("zero division")

    return res


def _get_def_stuff(npars):
    pars = np.zeros(npars) + PDEF
    cov = np.zeros((npars, npars)) + CDEF
    err = np.zeros(npars) + CDEF
    return pars, cov, err


def _test_cov(pcov):
    flags = 0
    try:
        e, v = np.linalg.eig(pcov)
        (weig,) = np.where(e < 0)
        if weig.size > 0:
            flags |= LM_NEG_COV_EIG

        (wneg,) = np.where(np.diag(pcov) < 0)
        if wneg.size > 0:
            flags |= LM_NEG_COV_DIAG

    except np.linalg.linalg.LinAlgError:
        flags |= EIG_NOTFINITE

    return flags


def _internal2external_grad(xi, bounds):
    """
    Calculate the internal (unconstrained) to external (constained)
    parameter gradiants.
    """
    grad = empty_like(xi)
    for i, (v, bound) in enumerate(zip(xi, bounds)):
        lower, upper = bound
        if lower is None and upper is None:  # No constraints
            grad[i] = 1.0
        elif upper is None:     # only lower bound
            grad[i] = v / sqrt(v * v + 1.)
        elif lower is None:     # only upper bound
            grad[i] = -v / sqrt(v * v + 1.)
        else:   # lower and upper bounds
            grad[i] = (upper - lower) * cos(v) / 2.
    return grad


def _internal2external_func(bounds):
    """
    Make a function which converts between internal (unconstrained) and
    external (constrained) parameters.
    """
    ls = [_internal2external_lambda(b) for b in bounds]

    def convert_i2e(xi):
        xe = empty_like(xi)
        xe[:] = [lam(p) for lam, p in zip(ls, xi)]
        return xe

    return convert_i2e


def _internal2external_lambda(bound):
    """
    Make a lambda function which converts a single internal (uncontrained)
    parameter to a external (constrained) parameter.
    """
    lower, upper = bound

    if lower is None and upper is None:  # no constraints
        return lambda x: x
    elif upper is None:     # only lower bound
        return lambda x: lower - 1. + sqrt(x * x + 1.)
    elif lower is None:     # only upper bound
        return lambda x: upper + 1. - sqrt(x * x + 1.)
    else:
        return lambda x: lower + ((upper - lower) / 2.) * (sin(x) + 1.)


def _external2internal_func(bounds):
    """
    Make a function which converts between external (constrained) and
    internal (unconstrained) parameters.
    """
    ls = [_external2internal_lambda(b) for b in bounds]

    def convert_e2i(xe):
        xi = empty_like(xe)
        xi[:] = [lam(p) for lam, p in zip(ls, xe)]
        return xi

    return convert_e2i


def _external2internal_lambda(bound):
    """
    Make a lambda function which converts an single external (constrained)
    parameter to a internal (unconstrained) parameter.
    """
    lower, upper = bound

    if lower is None and upper is None:  # no constraints
        return lambda x: x
    elif upper is None:     # only lower bound
        return lambda x: sqrt((x - lower + 1.) ** 2 - 1)
    elif lower is None:     # only upper bound
        return lambda x: sqrt((upper - x + 1.) ** 2 - 1)
    else:
        return lambda x: arcsin((2. * (x - lower) / (upper - lower)) - 1.)


def _check_func(checker, argname, thefunc, x0, args, numinputs,
                output_shape=None):
    res = atleast_1d(thefunc(*((x0[:numinputs],) + args)))
    if (output_shape is not None) and (shape(res) != output_shape):
        if (output_shape[0] != 1):
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = "%s: there is a mismatch between the input and output " \
                  "shape of the '%s' argument" % (checker, argname)
            func_name = getattr(thefunc, '__name__', None)
            if func_name:
                msg += " '%s'." % func_name
            else:
                msg += "."
            raise TypeError(msg)
    if issubdtype(res.dtype, inexact):
        dt = res.dtype
    else:
        dt = dtype(float)
    return shape(res), dt


def leastsqbound(func, x0, args=(), bounds=None, Dfun=None, full_output=0,
                 col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8,
                 gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    """
    Bounded minimization of the sum of squares of a set of equations.
    ::
        x = arg min(sum(func(y)**2,axis=0))
                 y
    Parameters
    ----------
    func : callable
        should take at least one (possibly length N vector) argument and
        returns M floating point numbers.
    x0 : ndarray
        The starting estimate for the minimization.
    args : tuple
        Any extra arguments to func are placed in this tuple.
    bounds : list
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction.
    Dfun : callable
        A function or method to compute the Jacobian of func with derivatives
        across the rows. If this is None, the Jacobian will be estimated.
    full_output : bool
        non-zero to return all optional outputs.
    col_deriv : bool
        non-zero to specify that the Jacobian function computes derivatives
        down the columns (faster, because there is no transpose operation).
    ftol : float
        Relative error desired in the sum of squares.
    xtol : float
        Relative error desired in the approximate solution.
    gtol : float
        Orthogonality desired between the function vector and the columns of
        the Jacobian.
    maxfev : int
        The maximum number of calls to the function. If zero, then 100*(N+1) is
        the maximum where N is the number of elements in x0.
    epsfcn : float
        A suitable step length for the forward-difference approximation of the
        Jacobian (for Dfun=None). If epsfcn is less than the machine precision,
        it is assumed that the relative errors in the functions are of the
        order of the machine precision.
    factor : float
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
    diag : sequence
        N positive entries that serve as a scale factors for the variables.
    Returns
    -------
    x : ndarray
        The solution (or the result of the last iteration for an unsuccessful
        call).
    cov_x : ndarray
        Uses the fjac and ipvt optional outputs to construct an
        estimate of the jacobian around the solution.  ``None`` if a
        singular matrix encountered (indicates very flat curvature in
        some direction).  This matrix must be multiplied by the
        residual standard deviation to get the covariance of the
        parameter estimates -- see curve_fit.
    infodict : dict
        a dictionary of optional outputs with the key s::
            - 'nfev' : the number of function calls
            - 'fvec' : the function evaluated at the output
            - 'fjac' : A permutation of the R matrix of a QR
                     factorization of the final approximate
                     Jacobian matrix, stored column wise.
                     Together with ipvt, the covariance of the
                     estimate can be approximated.
            - 'ipvt' : an integer array of length N which defines
                     a permutation matrix, p, such that
                     fjac*p = q*r, where r is upper triangular
                     with diagonal elements of nonincreasing
                     magnitude. Column j of p is column ipvt(j)
                     of the identity matrix.
            - 'qtf'  : the vector (transpose(q) * fvec).
    mesg : str
        A string message giving information about the cause of failure.
    ier : int
        An integer flag.  If it is equal to 1, 2, 3 or 4, the solution was
        found.  Otherwise, the solution was not found. In either case, the
        optional output variable 'mesg' gives more information.
    Notes
    -----
    "leastsq" is a wrapper around MINPACK's lmdif and lmder algorithms.
    cov_x is a Jacobian approximation to the Hessian of the least squares
    objective function.
    This approximation assumes that the objective function is based on the
    difference between some observed target data (ydata) and a (non-linear)
    function of the parameters `f(xdata, params)` ::
           func(params) = ydata - f(xdata, params)
    so that the objective function is ::
           min   sum((ydata - f(xdata, params))**2, axis=0)
         params
    Contraints on the parameters are enforced using an internal parameter list
    with appropiate transformations such that these internal parameters can be
    optimized without constraints. The transfomation between a given internal
    parameter, p_i, and a external parameter, p_e, are as follows:
    With ``min`` and ``max`` bounds defined ::
        p_i = arcsin((2 * (p_e - min) / (max - min)) - 1.)
        p_e = min + ((max - min) / 2.) * (sin(p_i) + 1.)
    With only ``max`` defined ::
        p_i = sqrt((max - p_e + 1.)**2 - 1.)
        p_e = max + 1. - sqrt(p_i**2 + 1.)
    With only ``min`` defined ::
        p_i = sqrt((p_e - min + 1.)**2 - 1.)
        p_e = min - 1. + sqrt(p_i**2 + 1.)
    These transfomations are used in the MINUIT package, and described in
    detail in the section 1.3.1 of the MINUIT User's Guide.
    When a parameter being optimized takes an on values near one of the
    imposed bounds the optimization can become blocked and the solution
    returned may be non-optimal.  In addition, near the limits cov_x and
    related returned variables do not contain meanful results.
    For best results, bounds of parameters should be limited to only those
    which are absolutly necessary, limits should be made wide enough to avoid
    parameters taking values near these limits and the optimization should be
    repeated without limits after a satisfactory minimum has been found for
    best error analysis. See section 1.3 and 5.3 of the MINUIT User's Guide
    for addition discussion of this topic.
    To Do
    -----
    Currently the ``factor`` and ``diag`` parameters scale the
    internal parameter list, but should scale the external parameter list.
    The `qtf` vector in the infodic dictionary reflects internal parameter
    list, it should be correct to reflect the external parameter list.
    References
    ----------
    * F. James and M. Winkler. MINUIT User's Guide, July 16, 2004.
    """
    # use leastsq if no bounds are present
    if bounds is None:
        return leastsq(func, x0, args, Dfun, full_output, col_deriv,
                       ftol, xtol, gtol, maxfev, epsfcn, factor, diag)

    # create function which convert between internal and external parameters
    i2e = _internal2external_func(bounds)
    e2i = _external2internal_func(bounds)

    x0 = asarray(x0).flatten()
    i0 = e2i(x0)
    n = len(x0)
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    if not isinstance(args, tuple):
        args = (args,)
    shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)
    m = shape[0]
    if n > m:
        raise TypeError('Improper input: N=%s must not exceed M=%s' % (n, m))
    if epsfcn is None:
        epsfcn = finfo(dtype).eps

    # define a wrapped func which accept internal parameters, converts them
    # to external parameters and calls func
    def wfunc(x, *args):
        return func(i2e(x), *args)

    if Dfun is None:
        if (maxfev == 0):
            maxfev = 200 * (n + 1)
        retval = _minpack._lmdif(wfunc, i0, args, full_output, ftol, xtol,
                                 gtol, maxfev, epsfcn, factor, diag)
    else:
        if col_deriv:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
        else:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (m, n))
        if (maxfev == 0):
            maxfev = 100 * (n + 1)

        def wDfun(x, *args):  # wrapped Dfun
            scale = _internal2external_grad(x, bounds)
            if col_deriv == 1:
                scale = scale.reshape(len(x), 1)
            return Dfun(i2e(x), *args)*scale

        retval = _minpack._lmder(wfunc, wDfun, i0, args, full_output,
                                 col_deriv, ftol, xtol, gtol, maxfev,
                                 factor, diag)

    errors = {0: ["Improper input parameters.", TypeError],
              1: ["Both actual and predicted relative reductions "
                  "in the sum of squares\n  are at most %f" % ftol, None],
              2: ["The relative error between two consecutive "
                  "iterates is at most %f" % xtol, None],
              3: ["Both actual and predicted relative reductions in "
                  "the sum of squares\n  are at most %f and the "
                  "relative error between two consecutive "
                  "iterates is at \n  most %f" % (ftol, xtol), None],
              4: ["The cosine of the angle between func(x) and any "
                  "column of the\n  Jacobian is at most %f in "
                  "absolute value" % gtol, None],
              5: ["Number of calls to function has reached "
                  "maxfev = %d." % maxfev, ValueError],
              6: ["ftol=%f is too small, no further reduction "
                  "in the sum of squares\n  is possible.""" % ftol,
                  ValueError],
              7: ["xtol=%f is too small, no further improvement in "
                  "the approximate\n  solution is possible." % xtol,
                  ValueError],
              8: ["gtol=%f is too small, func(x) is orthogonal to the "
                  "columns of\n  the Jacobian to machine "
                  "precision." % gtol, ValueError],
              'unknown': ["Unknown error.", TypeError]}

    info = retval[-1]    # The FORTRAN return value

    if (info not in [1, 2, 3, 4] and not full_output):
        if info in [5, 6, 7, 8]:
            warnings.warn(errors[info][0], RuntimeWarning)
        else:
            try:
                raise errors[info][1](errors[info][0])
            except KeyError:
                raise errors['unknown'][1](errors['unknown'][0])

    mesg = errors[info][0]
    x = i2e(retval[0])  # internal params to external params

    if full_output:
        # convert fjac from internal params to external
        grad = _internal2external_grad(retval[0], bounds)
        retval[1]['fjac'] = (retval[1]['fjac'].T / take(grad,
                             retval[1]['ipvt'] - 1)).T
        cov_x = None
        if info in [1, 2, 3, 4]:
            from numpy.dual import inv
            from numpy.linalg import LinAlgError
            perm = take(eye(n), retval[1]['ipvt'] - 1, 0)
            r = triu(transpose(retval[1]['fjac'])[:n, :])
            R = dot(r, perm)
            try:
                cov_x = inv(dot(transpose(R), R))
            except (LinAlgError, ValueError):
                pass
        return (x, cov_x) + retval[1:-1] + (mesg, info)
    else:
        return (x, info)

    
    
    
from numpy import zeros, exp, sqrt



class PriorSimpleSep(object):
    """
    Separate priors on each parameter
    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    T_prior:
        Prior on T or some size parameter
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self, cen_prior, g_prior, T_prior, F_prior):

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.T_prior = T_prior

        if isinstance(F_prior, list):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        allp = [self.T_prior] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_widths(self, nrand=10000):
        """
        estimate the width in each dimension
        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw
        """
        if not hasattr(self, "_sigma_estimates"):
            samples = self.sample(nrand)
            sigmas = samples.std(axis=0)

            # for e1,e2 we want to allow this a bit bigger
            # for very small objects.  Steps for MH could be
            # as large as half this
            sigmas[2] = 2.0
            sigmas[3] = 2.0

            self._sigma_estimates = sigmas

        return self._sigma_estimates

    def fill_fdiff(self, pars, fdiff):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        Parameters
        ----------
        pars: array
            Array of parameters values
        fdiff: array
            the fdiff array to fill
        """
        index = 0

        lnp1, lnp2 = self.cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1
        fdiff[index] = self.T_prior.get_lnprob_scalar(pars[4])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[5 + i])
            index += 1

        chi2 = -2 * fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index

    def get_prob_scalar(self, pars):
        """
        probability for scalar input (meaning one point)
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.get_lnprob_scalar(pars)
        p = exp(lnp)
        return p

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[5 + i])

        return lnp

    def get_prob_array(self, pars):
        """
        probability for array input [N,ndims]
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.get_lnprob_array(pars)
        p = exp(lnp)

        return p

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.T_prior.get_lnprob_array(pars[:, 4])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 5 + i])

        return lnp

    def sample(self, nrand=None):
        """
        Get random samples
        Parameters
        ----------
        nrand: int, optional
            Number of samples, default to a single set with size [npars].  If n
            is sent the result will have shape [n, npars]
        """

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        samples = zeros((nrand, 5 + self.nband))

        cen1, cen2 = self.cen_prior.sample(nrand)
        g1, g2 = self.g_prior.sample2d(nrand)
        T = self.T_prior.sample(nrand)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(nrand)
            samples[:, 5 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [str(self.cen_prior), str(self.g_prior), str(self.T_prior)]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorGalsimSimpleSep(PriorSimpleSep):
    """
    Separate priors on each parameter.  Wraps the T-based
    prior for ngmix models to provide a clear interface for
    r50
    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    r50_prior:
        Prior on T or some size parameter
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self, cen_prior, g_prior, r50_prior, F_prior):
        # re-use the name
        super().__init__(
            cen_prior=cen_prior,
            g_prior=g_prior,
            T_prior=r50_prior,
            F_prior=F_prior,
        )


class PriorBDSep(PriorSimpleSep):
    """
    Separate priors on each parameter
    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    T_prior:
        Prior on T or some size parameter
    logTratio:
        Prior on Td/Te
    fracdev_prior:
        Prior on fracdev for bulge+disk
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(
        self,
        cen_prior,
        g_prior,
        T_prior,
        logTratio_prior,
        fracdev_prior,
        F_prior,
    ):

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.T_prior = T_prior
        self.logTratio_prior = logTratio_prior
        self.fracdev_prior = fracdev_prior

        if isinstance(F_prior, (list, tuple)):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        allp = [
            self.T_prior,
            self.logTratio_prior,
            self.fracdev_prior,
        ] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])
        lnp += self.logTratio_prior.get_lnprob_scalar(pars[5])
        lnp += self.fracdev_prior.get_lnprob_scalar(pars[6])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[7 + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        (model-data)/err
        but "data" here is the central value of a prior.
        Parameters
        ----------
        pars: array
            Array of parameters values
        fdiff: array
            the fdiff array to fill
        """
        index = 0

        fdiff1, fdiff2 = self.cen_prior.get_fdiff(pars[0], pars[1])

        fdiff[index] = fdiff1
        index += 1
        fdiff[index] = fdiff2
        index += 1

        fdiff[index] = self.g_prior.get_fdiff(pars[2], pars[3])
        index += 1
        fdiff[index] = self.T_prior.get_fdiff(pars[4])
        index += 1

        fdiff[index] = self.logTratio_prior.get_fdiff(pars[5])
        index += 1

        fdiff[index] = self.fracdev_prior.get_fdiff(pars[6])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_fdiff(pars[7 + i])
            index += 1

        return index

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.T_prior.get_lnprob_array(pars[:, 4])
        lnp += self.logTratio_prior.get_lnprob_array(pars[:, 5])
        lnp += self.fracdev_prior.get_lnprob_array(pars[:, 6])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 7 + i])

        return lnp

    def sample(self, nrand=None):
        """
        Get random samples
        Parameters
        ----------
        nrand: int, optional
            Number of samples, default to a single set with size [npars].  If n
            is sent the result will have shape [n, npars]
        """

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        samples = zeros((nrand, 7 + self.nband))

        cen1, cen2 = self.cen_prior.sample(nrand)
        g1, g2 = self.g_prior.sample2d(nrand)
        T = self.T_prior.sample(nrand)
        logTratio = self.logTratio_prior.sample(nrand)
        fracdev = self.fracdev_prior.sample(nrand)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T
        samples[:, 5] = logTratio
        samples[:, 6] = fracdev

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(nrand)
            samples[:, 7 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [
            str(self.cen_prior),
            str(self.g_prior),
            str(self.T_prior),
            str(self.logTratio_prior),
            str(self.fracdev_prior),
        ]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorBDFSep(PriorSimpleSep):
    """
    Separate priors on each parameter
    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    T_prior:
        Prior on T or some size parameter
    fracdev_prior:
        Prior on fracdev for bulge+disk
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self, cen_prior, g_prior, T_prior, fracdev_prior, F_prior):

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.T_prior = T_prior
        self.fracdev_prior = fracdev_prior

        if isinstance(F_prior, (list, tuple)):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        allp = [self.T_prior, self.fracdev_prior] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])
        lnp += self.fracdev_prior.get_lnprob_scalar(pars[5])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[6 + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        (model-data)/err
        but "data" here is the central value of a prior.
        Parameters
        ----------
        pars: array
            Array of parameters values
        fdiff: array
            the fdiff array to fill
        """
        index = 0

        fdiff1, fdiff2 = self.cen_prior.get_fdiff(pars[0], pars[1])

        fdiff[index] = fdiff1
        index += 1
        fdiff[index] = fdiff2
        index += 1

        fdiff[index] = self.g_prior.get_fdiff(pars[2], pars[3])
        index += 1
        fdiff[index] = self.T_prior.get_fdiff(pars[4])
        index += 1

        fdiff[index] = self.fracdev_prior.get_fdiff(pars[5])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_fdiff(pars[6 + i])
            index += 1

        return index

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.T_prior.get_lnprob_array(pars[:, 4])
        lnp += self.fracdev_prior.get_lnprob_array(pars[:, 5])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 6 + i])

        return lnp

    def sample(self, nrand=None):
        """
        Get random samples
        Parameters
        ----------
        nrand: int, optional
            Number of samples, default to a single set with size [npars].  If n
            is sent the result will have shape [n, npars]
        """
        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        samples = zeros((nrand, 6 + self.nband))

        cen1, cen2 = self.cen_prior.sample(nrand)
        g1, g2 = self.g_prior.sample2d(nrand)
        T = self.T_prior.sample(nrand)
        fracdev = self.fracdev_prior.sample(nrand)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T
        samples[:, 5] = fracdev

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(nrand)
            samples[:, 6 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [
            str(self.cen_prior),
            str(self.g_prior),
            str(self.T_prior),
            str(self.fracdev_prior),
        ]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorSpergelSep(PriorSimpleSep):
    """
    Separate priors on each parameter of a Spergel profile
    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    r50_prior:
        Prior on r50
    nu_prior:
        Prior on the index nu
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self, cen_prior, g_prior, r50_prior, nu_prior, F_prior):

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.r50_prior = r50_prior
        self.nu_prior = nu_prior

        if isinstance(F_prior, list):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        allp = [self.r50_prior, self.nu_prior] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.r50_prior.get_lnprob_scalar(pars[4])
        lnp += self.nu_prior.get_lnprob_scalar(pars[5])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[6 + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        Parameters
        ----------
        pars: array
            Array of parameters values
        fdiff: array
            the fdiff array to fill
        """
        index = 0

        lnp1, lnp2 = self.cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1
        fdiff[index] = self.r50_prior.get_lnprob_scalar(pars[4])
        index += 1

        fdiff[index] = self.nu_prior.get_lnprob_scalar(pars[5])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[6 + i])
            index += 1

        chi2 = -2 * fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.r50_prior.get_lnprob_array(pars[:, 4])
        lnp += self.nu_prior.get_lnprob_array(pars[:, 5])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 6 + i])

        return lnp

    def sample(self, nrand=None):
        """
        Get random samples
        Parameters
        ----------
        nrand: int, optional
            Number of samples, default to a single set with size [npars].  If n
            is sent the result will have shape [n, npars]
        """

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        samples = zeros((nrand, 6 + self.nband))

        cen1, cen2 = self.cen_prior.sample(nrand)
        g1, g2 = self.g_prior.sample2d(nrand)
        r50 = self.r50_prior.sample(nrand)
        nu = self.nu_prior.sample(nrand)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = r50
        samples[:, 5] = nu

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(nrand)
            samples[:, 6 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [
            str(self.cen_prior),
            str(self.g_prior),
            str(self.r50_prior),
            str(self.nu_prior),
        ]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorCoellipSame(PriorSimpleSep):
    def __init__(self, ngauss, cen_prior, g_prior, T_prior, F_prior):

        self.ngauss = ngauss
        self.npars = gmix.get_coellip_npars(ngauss)

        super(PriorCoellipSame, self).__init__(
            cen_prior, g_prior, T_prior, F_prior
        )

        if self.nband != 1:
            raise ValueError("coellip only supports one band")

    def set_bounds(self):
        """
        set possibe bounds
        """

        ngauss = self.ngauss

        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        some_have_bounds = False

        allp = [self.T_prior]*ngauss + self.F_priors

        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                pbounds = [(p.bounds[0], p.bounds[1])]
            else:
                pbounds = [(None, None)]

            bounds += [pbounds] * self.ngauss

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        Parameters
        ----------
        pars: array
            Array of parameters values
        """

        if len(pars) != self.npars:
            raise ValueError('pars size %d expected %d' % (len(pars), self.npars))

        ngauss = self.ngauss

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])

        for i in range(ngauss):
            lnp += self.T_prior.get_lnprob_scalar(pars[4 + i])

        F_prior = self.F_priors[0]
        for i in range(ngauss):
            lnp += F_prior.get_lnprob_scalar(pars[4 + ngauss + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        Parameters
        ----------
        pars: array
            Array of parameters values
        fdiff: array
            the fdiff array to fill
        """

        if len(pars) != self.npars:
            raise ValueError('pars size %d expected %d' % (len(pars), self.npars))

        ngauss = self.ngauss

        index = 0

        lnp1, lnp2 = self.cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1

        for i in range(ngauss):
            fdiff[index] = self.T_prior.get_lnprob_scalar(pars[4 + i])
            index += 1

        F_prior = self.F_priors[0]
        for i in range(ngauss):
            fdiff[index] = F_prior.get_lnprob_scalar(
                pars[4 + ngauss + i]
            )
            index += 1

        chi2 = -2 * fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index

    def sample(self, nrand=None):
        """
        Get random samples
        Parameters
        ----------
        nrand: int, optional
            Number of samples, default to a single set with size [npars].  If n
            is sent the result will have shape [n, npars]
        """

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        ngauss = self.ngauss
        samples = zeros((nrand, self.npars))

        cen1, cen2 = self.cen_prior.sample(nrand)
        g1, g2 = self.g_prior.sample2d(nrand)
        T = self.T_prior.sample(nrand)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T

        for i in range(ngauss):
            samples[:, 4+i] += self.T_prior.sample(nrand)

        F_prior = self.F_priors[0]
        for i in range(ngauss):
            samples[:, 4 + ngauss + i] = F_prior.sample(nrand)

        if is_scalar:
            samples = samples[0, :]
        return samples
GMIX_LOW_DETVAL = 1.0e-200
from numpy import array

@njit
def gmix_eval_pixel_fast(gmix, pixel):
    """
    evaluate a single gaussian mixture, using the
    fast exponential
    """
    model_val = 0.0
    for igauss in range(gmix.size):

        model_val += gauss2d_eval_pixel_fast(gmix[igauss], pixel,)

    return model_val


@njit
def gauss2d_eval_pixel_fast(gauss, pixel):
    """
    evaluate a 2-d gaussian at the specified location, using
    the fast exponential
    parameters
    ----------
    gauss2d: gauss2d structure
        row,col,dcc,drr,drc,pnorm... See gmix.py
    v,u: numbers
        location in v,u plane (row,col for simple transforms)
    """
    model_val = 0.0

    # v->row, u->col in gauss
    vdiff = pixel["v"] - gauss["row"]
    udiff = pixel["u"] - gauss["col"]

    chi2 = (
        gauss["dcc"] * vdiff * vdiff
        + gauss["drr"] * udiff * udiff
        - 2.0 * gauss["drc"] * vdiff * udiff
    )

    if chi2 < FASTEXP_MAX_CHI2 and chi2 >= 0.0:
        model_val = gauss["pnorm"] * fexp(-0.5 * chi2) * pixel["area"]

    return model_val


@njit
def gauss2d_eval_pixel(gauss, pixel):
    """
    evaluate a 2-d gaussian at the specified location
    parameters
    ----------
    gauss2d: gauss2d structure
        row,col,dcc,drr,drc,pnorm... See gmix.py
    pixel: struct with coods
        should have fields v,u
    """
    model_val = 0.0

    # v->row, u->col in gauss
    vdiff = pixel["v"] - gauss["row"]
    udiff = pixel["u"] - gauss["col"]

    chi2 = (
        gauss["dcc"] * vdiff * vdiff
        + gauss["drr"] * udiff * udiff
        - 2.0 * gauss["drc"] * vdiff * udiff
    )

    model_val = gauss["pnorm"] * numpy.exp(-0.5 * chi2) * pixel["area"]

    return model_val


@njit
def gmix_eval_pixel(gmix, pixel):
    """
    evaluate a single gaussian mixture
    """
    model_val = 0.0
    for igauss in range(gmix.size):

        model_val += gauss2d_eval_pixel(gmix[igauss], pixel,)

    return model_val


@njit
def gmix_get_cen(gmix):
    """
    get the center of the gaussian mixture, as well as
    the psum
    """
    row = 0.0
    col = 0.0
    psum = 0.0

    n_gauss = gmix.size
    for i in range(n_gauss):
        gauss = gmix[i]

        p = gauss["p"]
        row += p * gauss["row"]
        col += p * gauss["col"]
        psum += p

    row /= psum
    col /= psum

    return row, col, psum


@njit
def gmix_get_e1e2T(gmix):
    """
    get e1,e2,T for the gaussian mixture
    """

    row, col, psum0 = gmix_get_cen(gmix)

    if psum0 == 0.0:
        raise GMixRangeError("cannot calculate T due to zero psum")

    n_gauss = gmix.size
    psum = 0.0
    irr_sum = 0.0
    irc_sum = 0.0
    icc_sum = 0.0

    for i in range(n_gauss):
        gauss = gmix[i]

        p = gauss["p"]

        rowdiff = gauss["row"] - row
        coldiff = gauss["col"] - col

        irr_sum += p * (gauss["irr"] + rowdiff * rowdiff)
        irc_sum += p * (gauss["irc"] + rowdiff * coldiff)
        icc_sum += p * (gauss["icc"] + coldiff * coldiff)

        psum += p

    T_sum = irr_sum + icc_sum
    T = T_sum / psum

    if T_sum <= 0.0:
        raise GMixRangeError("T <= 0.0")

    e1 = (icc_sum - irr_sum) / T_sum
    e2 = 2.0 * irc_sum / T_sum

    return e1, e2, T


@njit
def gmix_set_norms(gmix):
    """
    set all norms for gaussians in the input gaussian mixture
    parameters
    ----------
    gmix:
       gaussian mixture
    """
    for gauss in gmix:
        gauss2d_set_norm(gauss)


@njit
def gauss2d_set_norm(gauss):
    """
    set the normalization, and nromalized variances
    A GMixRangeError is raised if the determinant is too small
    parameters
    ----------
    gauss: a 2-d gaussian structure
        See gmix.py
    """

    if gauss["det"] < GMIX_LOW_DETVAL:
        raise GMixRangeError("det too low")

    T = gauss["irr"] + gauss["icc"]
    if T <= GMIX_LOW_DETVAL:
        raise GMixRangeError("T too low")

    idet = 1.0 / gauss["det"]
    gauss["drr"] = gauss["irr"] * idet
    gauss["drc"] = gauss["irc"] * idet
    gauss["dcc"] = gauss["icc"] * idet
    gauss["norm"] = 1.0 / (2 * numpy.pi * numpy.sqrt(gauss["det"]))

    gauss["pnorm"] = gauss["p"] * gauss["norm"]

    gauss["norm_set"] = 1


@njit
def gauss2d_set(gauss, p, row, col, irr, irc, icc):
    """
    set the gaussian, clearing normalizations
    """
    gauss["norm_set"] = 0
    gauss["drr"] = nan
    gauss["drc"] = nan
    gauss["dcc"] = nan
    gauss["norm"] = nan
    gauss["pnorm"] = nan

    gauss["p"] = p
    gauss["row"] = row
    gauss["col"] = col
    gauss["irr"] = irr
    gauss["irc"] = irc
    gauss["icc"] = icc

    gauss["det"] = irr * icc - irc * irc


_pvals_exp = array(
    [
        0.00061601229677880041,
        0.0079461395724623237,
        0.053280454055540001,
        0.21797364640726541,
        0.45496740582554868,
        0.26521634184240478,
    ]
)

_fvals_exp = array(
    [
        0.002467115141477932,
        0.018147435573256168,
        0.07944063151366336,
        0.27137669897479122,
        0.79782256866993773,
        2.1623306025075739,
    ]
)

_pvals_dev = array(
    [
        6.5288960012625658e-05,
        0.00044199216814302695,
        0.0020859587871659754,
        0.0075913681418996841,
        0.02260266219257237,
        0.056532254390212859,
        0.11939049233042602,
        0.20969545753234975,
        0.29254151133139222,
        0.28905301416582552,
    ]
)

_fvals_dev = array(
    [
        2.9934935706271918e-07,
        3.4651596338231207e-06,
        2.4807910570562753e-05,
        1.4307404300535354e-04,
        7.2753169298239500e-04,
        3.4582464394427260e-03,
        1.6086645440719100e-02,
        7.7006776775654429e-02,
        4.1012562102501476e-01,
        2.9812509778548648e00,
    ]
)

_pvals_turb = array(
    [0.596510042804182, 0.4034898268889178, 1.303069003078001e-07]
)

_fvals_turb = array(
    [0.5793612389470884, 1.621860687127999, 7.019347162356363]
)

_pvals_gauss = array([1.0])
_fvals_gauss = array([1.0])


@njit
def gmix_fill_simple(gmix, pars, fvals, pvals):
    """
    fill a simple (6 parameter) gaussian mixture model
    no error checking done here
    """

    row = pars[0]
    col = pars[1]
    g1 = pars[2]
    g2 = pars[3]
    T = pars[4]
    flux = pars[5]

    e1, e2 = g1g2_to_e1e2(g1, g2)

    n_gauss = gmix.size
    for i in range(n_gauss):

        gauss = gmix[i]

        T_i_2 = 0.5 * T * fvals[i]
        flux_i = flux * pvals[i]

        gauss2d_set(
            gauss,
            flux_i,
            row,
            col,
            T_i_2 * (1 - e1),
            T_i_2 * e2,
            T_i_2 * (1 + e1),
        )


@njit
def gmix_fill_exp(gmix, pars):
    """
    fill an exponential model
    """
    gmix_fill_simple(gmix, pars, _fvals_exp, _pvals_exp)


@njit
def gmix_fill_dev(gmix, pars):
    """
    fill a dev model
    """
    gmix_fill_simple(gmix, pars, _fvals_dev, _pvals_dev)


@njit
def gmix_fill_turb(gmix, pars):
    """
    fill a turbulent psf model
    """
    gmix_fill_simple(gmix, pars, _fvals_turb, _pvals_turb)


@njit
def gmix_fill_gauss(gmix, pars):
    """
    fill a gaussian model
    """
    gmix_fill_simple(gmix, pars, _fvals_gauss, _pvals_gauss)


@njit
def gmix_fill_coellip(gmix, pars):
    """
    fill a coelliptical model
    [cen1,cen2,g1,g2,T1,T2,...,F1,F2...]
    """

    row = pars[0]
    col = pars[1]
    g1 = pars[2]
    g2 = pars[3]

    e1, e2 = g1g2_to_e1e2(g1, g2)

    n_gauss = gmix.size

    for i in range(n_gauss):
        T = pars[4 + i]
        Thalf = 0.5 * T
        flux = pars[4 + n_gauss + i]

        gauss2d_set(
            gmix[i],
            flux,
            row,
            col,
            Thalf * (1 - e1),
            Thalf * e2,
            Thalf * (1 + e1),
        )


@njit
def gmix_fill_full(gmix, pars):
    """
    fill a "full" gmix model, parameters are specified
    for each gaussian independently
    """

    n_gauss = gmix.size
    for i in range(n_gauss):
        beg = i * 6

        gauss2d_set(
            gmix[i],
            pars[beg + 0],
            pars[beg + 1],
            pars[beg + 2],
            pars[beg + 3],
            pars[beg + 4],
            pars[beg + 5],
        )


@njit
def gmix_fill_cm(gmix, fracdev, TdByTe, Tfactor, pars):
    """
    fill a composite model
    """

    row = pars[0]
    col = pars[1]
    g1 = pars[2]
    g2 = pars[3]
    T = pars[4] * Tfactor
    flux = pars[5]

    ifracdev = 1.0 - fracdev

    e1, e2 = g1g2_to_e1e2(g1, g2)

    for i in range(16):
        if i < 6:
            p = _pvals_exp[i] * ifracdev
            f = _fvals_exp[i]
        else:
            p = _pvals_dev[i - 6] * fracdev
            f = _fvals_dev[i - 6] * TdByTe

        T_i_2 = 0.5 * T * f
        flux_i = flux * p

        gauss2d_set(
            gmix[i],
            flux_i,
            row,
            col,
            T_i_2 * (1 - e1),
            T_i_2 * e2,
            T_i_2 * (1 + e1),
        )


@njit
def gmix_fill_bd(gmix, pars):
    """
    fill a bulge plus disk model
    """

    row = pars[0]
    col = pars[1]
    g1 = pars[2]
    g2 = pars[3]
    T = pars[4]
    lTrat = pars[5]
    fracdev = pars[6]
    flux = pars[7]

    TdByTe = 10.0 ** lTrat

    Tfactor = get_cm_Tfactor(fracdev, TdByTe)
    T = T * Tfactor

    ifracdev = 1.0 - fracdev

    e1, e2 = g1g2_to_e1e2(g1, g2)

    for i in range(16):
        if i < 6:
            p = _pvals_exp[i] * ifracdev
            f = _fvals_exp[i]
        else:
            p = _pvals_dev[i - 6] * fracdev
            f = _fvals_dev[i - 6] * TdByTe

        T_i_2 = 0.5 * T * f
        flux_i = flux * p

        gauss2d_set(
            gmix[i],
            flux_i,
            row,
            col,
            T_i_2 * (1 - e1),
            T_i_2 * e2,
            T_i_2 * (1 + e1),
        )


@njit
def gmix_fill_bdf(gmix, pars):
    """
    fill a composite model with fixed Td/Te=1 but fracdev
    varying
    """

    TdByTe = 1.0

    row = pars[0]
    col = pars[1]
    g1 = pars[2]
    g2 = pars[3]
    T = pars[4]
    fracdev = pars[5]
    flux = pars[6]

    Tfactor = get_cm_Tfactor(fracdev, TdByTe)
    T = T * Tfactor

    ifracdev = 1.0 - fracdev

    e1, e2 = g1g2_to_e1e2(g1, g2)

    for i in range(16):
        if i < 6:
            p = _pvals_exp[i] * ifracdev
            f = _fvals_exp[i]
        else:
            p = _pvals_dev[i - 6] * fracdev
            f = _fvals_dev[i - 6] * TdByTe

        T_i_2 = 0.5 * T * f
        flux_i = flux * p

        gauss2d_set(
            gmix[i],
            flux_i,
            row,
            col,
            T_i_2 * (1 - e1),
            T_i_2 * e2,
            T_i_2 * (1 + e1),
        )


@njit
def get_cm_Tfactor(fracdev, TdByTe):
    """
    get the factor needed to convert T to the T needed
    for using in filling a cmodel gaussian mixture
    parameters
    ----------
    fracdev: float
        fraction of flux in the dev component
    TdByTe: float
        T_{dev}/T_{exp}
    """

    ifracdev = 1.0 - fracdev

    Tfactor = 0.0

    for i in range(6):
        p = _pvals_exp[i] * ifracdev
        f = _fvals_exp[i]

        Tfactor += p * f

    for i in range(10):
        p = _pvals_dev[i] * fracdev
        f = _fvals_dev[i] * TdByTe

        Tfactor += p * f

    Tfactor = 1.0 / Tfactor

    return Tfactor


_gmix_fill_functions = {
    "exp": gmix_fill_exp,
    "dev": gmix_fill_dev,
    "turb": gmix_fill_turb,
    "gauss": gmix_fill_gauss,
    "cm": gmix_fill_cm,
    "bd": gmix_fill_bd,
    "bdf": gmix_fill_bdf,
    "coellip": gmix_fill_coellip,
    "full": gmix_fill_full,
}


@njit
def gmix_convolve_fill(self, gmix, psf):
    """
    fill the gaussian mixture with the convolution of gmix0,
    the unconvolved mixture, and the psf
    parameters
    ----------
    self: gaussian mixture
        The convolved mixture, to be filled
    gmix: gaussian mixture
        The unconvolved mixture
    psf: gaussian mixture
        The psf with which to convolve
    """

    psf_rowcen, psf_colcen, psf_psum = gmix_get_cen(psf)

    psf_ipsum = 1.0 / psf_psum
    n_gauss = gmix.size
    psf_n_gauss = psf.size

    itot = 0
    for iobj in range(n_gauss):
        obj_gauss = gmix[iobj]

        for ipsf in range(psf_n_gauss):
            psf_gauss = psf[ipsf]

            p = obj_gauss["p"] * psf_gauss["p"] * psf_ipsum

            row = obj_gauss["row"] + (psf_gauss["row"] - psf_rowcen)
            col = obj_gauss["col"] + (psf_gauss["col"] - psf_colcen)

            irr = obj_gauss["irr"] + psf_gauss["irr"]
            irc = obj_gauss["irc"] + psf_gauss["irc"]
            icc = obj_gauss["icc"] + psf_gauss["icc"]

            gauss2d_set(self[itot], p, row, col, irr, irc, icc)

            itot += 1


@njit
def g1g2_to_e1e2(g1, g2):
    """
    convert g to e
    """

    g = numpy.sqrt(g1 * g1 + g2 * g2)

    if g >= 1:
        raise GMixRangeError("g >= 1")

    if g == 0.0:
        e1 = 0.0
        e2 = 0.0
    else:

        eta = 2 * numpy.arctanh(g)
        e = numpy.tanh(eta)
        if e >= 1.0:
            e = 0.99999999

        fac = e / g

        e1 = fac * g1
        e2 = fac * g2

    return e1, e2


@njit
def get_weighted_sums(wt, pixels, res, maxrad):
    """
    do sums for calculating the weighted moments
    """

    maxrad2 = maxrad ** 2

    vcen = wt["row"][0]
    ucen = wt["col"][0]
    F = res["F"]

    n_pixels = pixels.size
    for i_pixel in range(n_pixels):

        pixel = pixels[i_pixel]

        vmod = pixel["v"] - vcen
        umod = pixel["u"] - ucen

        rad2 = umod * umod + vmod * vmod
        if rad2 < maxrad2:

            weight = gmix_eval_pixel(wt, pixel)
            var = 1.0 / (pixel["ierr"] * pixel["ierr"])

            wdata = weight * pixel["val"]
            w2 = weight * weight

            F[0] = pixel["v"]
            F[1] = pixel["u"]
            F[2] = umod * umod - vmod * vmod
            F[3] = 2 * vmod * umod
            F[4] = rad2
            F[5] = 1.0

            res["wsum"] += weight
            res["npix"] += 1

            for i in range(6):
                res["sums"][i] += wdata * F[i]
                for j in range(6):
                    res["sums_cov"][i, j] += w2 * var * F[i] * F[j]


@njit
def get_loglike(gmix, pixels):
    """
    get the log likelihood
    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    returns
    -------
    a tuple of
    loglike: float
        log likelihood
    s2n_numer: float
        numerator for s/n
    s2n_demon: float
        will use sqrt(s2n_denom) for denominator for s/n
    npix: int
        number of pixels used
    """

    if gmix["norm_set"][0] == 0:
        gmix_set_norms(gmix)

    npix = 0
    loglike = s2n_numer = s2n_denom = 0.0

    n_pixels = pixels.shape[0]
    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel_fast(gmix, pixel)

        ivar = pixel["ierr"] * pixel["ierr"]
        val = pixel["val"]
        diff = model_val - val

        loglike += diff * diff * ivar

        s2n_numer += val * model_val * ivar
        s2n_denom += model_val * model_val * ivar
        npix += 1

    loglike *= -0.5

    return loglike, s2n_numer, s2n_denom, npix


@njit
def fill_fdiff(gmix, pixels, fdiff, start):
    """
    fill fdiff array (model-data)/err
    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    fdiff: array
        Array to fill, should be same length as pixels
    """

    if gmix["norm_set"][0] == 0:
        gmix_set_norms(gmix)

    n_pixels = pixels.shape[0]
    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel_fast(gmix, pixel)
        fdiff[start + ipixel] = (model_val - pixel["val"]) * pixel["ierr"]


@njit
def get_model_s2n_sum(gmix, pixels):
    """
    get the model s/n sum.
    The s/n is then sqrt(s2n_sum)
    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    returns
    -------
    s2n_sum: float
        sum to calculate s/n
    """

    if gmix["norm_set"][0] == 0:
        gmix_set_norms(gmix)

    n_pixels = pixels.shape[0]
    s2n_sum = 0.0

    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel_fast(gmix, pixel)
        ivar = pixel["ierr"] * pixel["ierr"]

        s2n_sum += model_val * model_val * ivar

    return s2n_sum
FASTEXP_MAX_CHI2 = 25.0

@njit
def exp5(x):
    """
    fast exponential
    in the range -15, 0 the relative error is at worst about -4.0e-5
    no range checking is done here, do it at the caller
    Parameters
    ----------
    x: number
        a number.  You should check it is in the valid range for
        the lookup table
    """

    ival = int(x-0.5)
    f = x - ival
    index = ival - _EXP_I0
    expval = _EXP_LOOKUP[index]
    expval *= (120+f*(120+f*(60+f*(20+f*(5+f)))))*0.0083333333

    return expval


fexp = exp5



def _make_exp_lookup(minval=-15, maxval=0):
    """
    lookup array in range [minval,0] inclusive
    """
    nlook = abs(maxval-minval)+1
    expvals = np.zeros(nlook, dtype='f8')

    ivals = np.arange(minval, maxval+1, dtype='i4')

    expvals[:] = np.exp(ivals)

    return ivals, expvals


FASTEXP_MAX_CHI2 = 25.0

# we limit to chi squared of 25, which means an argument of
# -0.5*25. Use -15 to be safe
_EXP_IVALS, _EXP_LOOKUP = _make_exp_lookup(
    minval=-15,
    maxval=0,
)
_EXP_I0 = _EXP_IVALS[0]
from numpy import nan



# always and forever
MAGZP_REF = 30.0

CONFIG = {
    'metacal': {
        # check for an edge hit
        'bmask_flags': 2**30,

        'metacal_pars': {
            'psf': 'fitgauss',
            'types': ['noshear', '1p', '1m', '2p', '2m'],
        },

        'model': 'gauss',

        'max_pars': {
            'ntry': 2,
            'pars': {
                'method': 'lm',
                'lm_pars': {
                    'maxfev': 2000,
                    'xtol': 5.0e-5,
                    'ftol': 5.0e-5,
                }
            }
        },

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.263
            },

            'g': {
                'type': 'ba',
                'sigma': 0.2
            },

            'T': {
                'type': 'two-sided-erf',
                'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
            },

            'flux': {
                'type': 'two-sided-erf',
                'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
            }
        },

        'psf': {
            'model': 'gauss',
            'ntry': 2,
            'lm_pars': {
                'maxfev': 2000,
                'ftol': 1.0e-5,
                'xtol': 1.0e-5
            }
        }
    },
}