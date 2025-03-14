"""
Matched Filter Abstract Base Class
"""
import abc
import time
import argparse
from enum import Enum

import pandas as pd

# numpy group imports
import numpy as np
from numpy.random import permutation
from numpy import cov as _cov

# scipy group
from scipy import ndimage
from scipy.linalg import LinAlgError
from scipy.linalg import inv as _inv
from scipy.linalg import eig as _eig
from scipy.linalg import det as _det


class MatchedFilter():

    """
    Implements shared functionality of matched filter algorithm developed by NASA to detect 
    CH4 / CO2 emissions in AVIRIS hyperspectral imagery as part of its Methane Source Finder 
    programme. 
    """

    class OutputBand(Enum):
        RED         =  (0, 'Red Radiance (uW/nm/sr/cm2)')
        GREEN       =  (1, 'Green Radiance (uW/nm/sr/cm2)')
        BLUE        =  (2, 'Blue Radiance (uW/nm/sr/cm2)')
        XCH4        =  (3, 'CH4 Absorption (ppm x m)')
        XCH4_MEDIAN =  (4, 'Median CH4 Absorption (ppm x m)')

        def __init__(self, value, description):
            # constructor
            self._value_ = value
            self._description_ = description

        @property
        def description(self):
            """
            return description
            """
            return self._description_


    # convert likelihood scores to ppm per m2
    _ppmscaling = 100000.0
    def __init__( self, args ):

        """
        constructor        
        """

        # get args
        self._args = args

        # alphas for leave-one-out cross validation shrinkage
        astep, aminexp, amaxexp = 0.05, -10.0, 0.0
        self._alphas = 10.0 ** np.arange( aminexp, amaxexp + astep, astep )

        # preallocate likelihood score vector
        self._nll = np.zeros( len( self._alphas ) )

        # exclude nonfinite + negative spectra in covariance estimates
        self.get_valid_index = lambda data: \
                        np.where(((~( data < 0 ) ) & np.isfinite( data ) ).all(axis=1))[ 0 ]


    @abc.abstractmethod
    def process( self ):

        """
        entry point function to override
        """
        return


    def get_column_data( self, img_data, idx ):

        """
        Get active channel data         
        """

        # get active channel data
        return img_data[ :, self._args.active_channels, idx ]


    def get_target_enhancement( self, img_data, target, out_data, nodata=-9999 ):

        """
        compute target enhancement data using matched filter         
        """

        # set memorymap to nodata
        outband = MatchedFilter.OutputBand
        out_data[:,:,outband.XCH4.value] = nodata

        # initialise stats
        ncols = img_data.shape[ -1 ]
        stats = {   'avg' : np.ones( ncols ) * nodata,
                    'std' : np.ones( ncols ) * nodata,
                    'count' : np.ones( ncols ) * nodata
        }

        # iterate through columns
        stime = time.time()
        print( f'Starting columnwise processing ({ncols} columns, {len( target )} channels)' )
        for idx in range( ncols ):

            # get 2D row x channel array indexed to column
            coldata = img_data[ :, target.index_lut.values, idx ]

            # copy data from rgb channels into output
            out_data[ :, idx, outband.RED.value ] = img_data[ :, self._args.rgb_bands[0], idx ]
            out_data[ :, idx, outband.GREEN.value ] = img_data[ :, self._args.rgb_bands[1], idx ]
            out_data[ :, idx, outband.BLUE.value ] = img_data[ :, self._args.rgb_bands[2], idx ]

            # extract valid data from current column
            valid_index = self.get_valid_index( coldata )
            valid_coldata = np.float64( coldata[ valid_index, : ].copy() )

            # proceed onto next column if no valid data in column
            if valid_coldata.size == 0 :
                continue

            # mean normalise column radiances
            mu = np.mean( valid_coldata, axis=0 )
            valid_coldata_norm = valid_coldata - mu

            try:

                # compute inverted covariance matrix using shrinkage estimator
                covar, _ = self.get_covariance_matrix(  valid_coldata_norm,
                                                    self._alphas,
                                                    self._nll )
                covar_inv = MatchedFilter.inv( covar )

            except LinAlgError:
                print( f'... skipping column {idx} - singular covariance matrix detected' )
                out_data[ :, idx, outband.XCH4.value ] = 0.0
                continue

            # normalise target spectra
            profile = target.absorbance.values.copy()
            profile = profile * mu

            # compute classical matched filter between data and target
            normalizer = profile.dot( covar_inv ).dot( profile.T )
            mf = ( valid_coldata_norm.dot( covar_inv ).dot( profile.T ) ) / normalizer
            mf = mf * MatchedFilter._ppmscaling

            # copy matched filter results into output memorymap (pixelwise units ppm x meter)
            out_data[ valid_index, idx, outband.XCH4.value ] = mf

            # update statistics
            stats[ 'std' ][ idx ] = np.std(mf)
            stats[ 'avg' ][ idx ] = np.mean(mf)
            stats[ 'count' ][ idx ] = mf.size

            # writeback results
            print(  f'column {idx} -'
                    f'mean: {stats[ "avg" ][ idx ]}, '
                    f'std: {stats[ "std" ][ idx ]}, '
                    f'count: {stats[ "count" ][ idx ]}'
            )

        # generate median filtered version of CH4 enhancement image
        out_data[ :, :, outband.XCH4_MEDIAN.value ]  \
                    = ndimage.median_filter( out_data[ :, :, outband.XCH4.value ], size=(7,7) )

        # replicate nodata mask
        out_data[ :, :, outband.XCH4_MEDIAN.value ] \
                    = np.where( out_data[ :, :, outband.XCH4.value ] == nodata, \
                                nodata, \
                                out_data[ :, :, outband.XCH4_MEDIAN.value ] )

        # replicate nodata mask
        out_data[ :, :, outband.XCH4_MEDIAN.value ] \
                    = np.where( out_data[ :, :, outband.XCH4_MEDIAN.value ] < 0,
                               0, \
                               out_data[ :, :, outband.XCH4_MEDIAN.value ] )

        # return stats
        print( f'... done (elapsed time={time.time() - stime}s)' )
        return stats


    @staticmethod
    def open_target_profile( pathname, active_channels ):

        """
        Load absorbance values of target spectral profile from text file
        """

        # load the gas spectrum
        df = pd.read_csv( pathname,
                          header=None,
                          delim_whitespace=True,
                          names=['channel_index', 'wavelength', 'absorbance']
        )
        return df [ df.channel_index.isin( active_channels ) ]


    @staticmethod
    def get_covariance_matrix( data, alphas, nll ):

        """    
        Implementation of the 'incredible shrinking covariance estimator' - repurposed from 
        source code taken from Methane Source Finder GitHub reporsitory: 
        https://github.com/dsmbgu8/srcfinder.
        
        To distinguish target from background requires that the background be well-characterized.
        When the background is modelled by a (global or local) Gaussian, a covariance matrix 
        must be estimated. The standard sample covariance overfits the data, and when the 
        training sample size is small, the target detection performance suffers.

        Shrinkage addresses the problem of overfitting that inevitably arises when a 
        high-dimensional model is fit from a small dataset. In place of the (overfit) sample 
        covariance matrix, a linear combination of that covariance with a fixed matrix is employed. 
        The fixed matrix might be the identity, the diagonal elements of the sample covariance, 
        or some other underfit estimator. A combination of an overfit with an underfit estimator 
        can lead to a well-fit estimator. 

        The coefficient defining relative contribution of overfit and underfit estimator is known 
        as the shrinkage parameter. It is generally estimated by some kind of cross-validation 
        approach which is typically computationally expensive. Extending work by Hoffbeck and 
        Landgrebe, this algorithm computes efficient approximation of the leave-one-out 
        cross-validation (LOOC) estimate of the shrinkage parameter - enabling accurate modelling of
        covariance matrix from a limited sample of data.

        The Thiel-based estimation of covariance matrix may be expressed as: 
                            Rα = (1 − α)S + αT
        where α is the shrinkage parameter, S is the overfit estimator (sample covariance) and 
        T is the underfit estimator.

        Link to paper: https://public.lanl.gov/jt/Papers/shrink-post-SPIE8391.pdf
        """

        # loocv shrinkage estimation via Theiler et al.
        stability_scaling = 100.0

        # get dims of input data and pre-compute Gaussian constant
        nsamples, nchan = data.shape
        nchanlog2pi = nchan * np.log( 2.0 * np.pi )

        # compute sample coveriance matrix (overfit) and identity matrix (underfit)
        X = data * stability_scaling
        S = MatchedFilter.cov(X)
        T = np.diag(np.diag(S))

        # initialise likelihood score vector
        nll[:] = np.inf

        # closed form for leave one out cross validation error
        for i, alpha in enumerate( alphas ):
            try:

                # compute shrinkage parameter
                beta = (1.0 - alpha) / (nsamples - 1.0)
                g_alpha = nsamples * (beta*S) + (alpha*T)

                # compute weighted determinant - check for singular matrix
                g_det = MatchedFilter.det(g_alpha)
                if g_det==0:
                    continue

                # compute likelihood of observing xk, given Rα,k = (1 − α)Sk + αT
                r_k  = (X.dot( MatchedFilter.inv(g_alpha)) * X).sum(axis=1)
                q = 1.0 - beta * r_k

                nll[i] = 0.5 * (nchanlog2pi + np.log(g_det)) +1.0 / (2.0*nsamples) * \
                                (np.log(q)+(r_k/q)).sum()

            except LinAlgError:
                print('LinAlgError exception error')

        # get index of likeliest result
        mindex = np.argmin(nll)

        # retrieve alpha value of likeliest result
        if nll[mindex]!=np.inf:
            alpha = alphas[mindex]
        else:
            # invalid result - revert to sample covariance
            mindex = -1
            alpha = 0.0

        # compute nonregularized covariance and shrinkage target
        S = MatchedFilter.cov( data )
        T = np.diag(np.diag(S))

        # compute covariance matrix
        C = (1.0 - alpha) * S + alpha * T
        return C, mindex


    @staticmethod
    def inv( A, **kwargs ):

        """
        compute matrix inverse 
        """

        # compute matrix inverse
        kwargs.setdefault('overwrite_a',False)
        kwargs.setdefault('check_finite',False)
        return _inv(A,**kwargs)


    @staticmethod
    def randperm( *args ):

        """
        compute random permutation 
        """

        # compute random permutation
        n = args[0]
        k = n if len(args) < 2 else args[1]
        return permutation(n)[:k]


    @staticmethod
    def cov( A, **kwargs ):

        """
        computes covariance matrix for n x m array of n samples with m features per sample
        """

        # compute covariance matrix
        kwargs.setdefault('ddof',1)
        return _cov(A.T,**kwargs)


    @staticmethod
    def eig( A, **kwargs ):

        """
        computes eigenvectors and eigenvalues
        """

        # compute eigenvectors and eigenvalues
        kwargs.setdefault('overwrite_a',False)
        kwargs.setdefault('check_finite',False)
        kwargs.setdefault('left',False)
        kwargs.setdefault('right',True)
        return _eig(A,**kwargs)


    @staticmethod
    def det( A,**kwargs ):

        """
        compute matrix determinant
        """

        # compute matrix determinant
        kwargs.setdefault('overwrite_a',False)
        kwargs.setdefault('check_finite',False)
        return _det(A,**kwargs)


    @staticmethod
    def parse_arguments():

        """
        parse command line arguments
        """

        # parse command line arguments
        parser = argparse.ArgumentParser( description='Robust Columnwise Matched Filter' )

        # input pathname
        parser.add_argument( 'input_pathname',
                             type=str,
                             help='pathname to hyperspectral image'
        )

        # target spectra pathname
        parser.add_argument( 'target_pathname',
                            type=str,
                            help='pathname to target spectral profile'
        )

        # output pathname
        parser.add_argument( 'out_pathname',
                            type=str,
                            help='pathname for output image (mf ch4 ppm)'
        )

        # optional args - rgb bands
        parser.add_argument('--rgb_bands',
                            default=[ 60,42,24 ],
                            help='RGB channel indices',
                            nargs=3,
                            type=int
        )

        # active channels
        parser.add_argument('--active_channels',
                            default=['352-423'],
                            help='comma-separated list of active channels',
                            type=lambda x: x.split(',')
        )

        # sampling rate
        parser.add_argument('--sampling_rate',
                            default=1,
                            help='channel sampling rate (test/validation)',
                            type=int
        )

        return parser.parse_args()


    @staticmethod
    def get_active_channels( args ):

        """
        get active channels
        """

        # iterate through comma separated substrings
        channels = []
        for arg in args.active_channels:

            # split hyphen by filling in values between start and end
            tokens = arg.split( '-' )
            if len( tokens ) == 2:
                channels.extend( list( range( int( tokens[ 0 ] ), int( tokens[ 1 ] ) + 1 )  ) )
            else:
                channels.append( int ( arg ) )

        return np.array( channels )
