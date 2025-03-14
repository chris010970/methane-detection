import os
import re
import glob
from datetime import datetime

import rasterio
import numpy as np
import pandas as pd

from base import MatchedFilter

class Ahsi( MatchedFilter ):

    class Image():

        def __init__( self, path, waveband ):

            """
            constructor
            """

            # open image
            filename = glob.glob( os.path.join( path, f'*_{waveband}.tiff' ) )[ 0 ]
            self.src = rasterio.open( filename , driver='GTiff' )
            self.transformer = rasterio.transform.RPCTransformer( self.src.rpcs )

            # radiance calibration coefficients
            filename = glob.glob( os.path.join( path, f'*_{waveband}IR_RadCal.raw' ) )[ 0 ]
            self.cal = pd.read_csv( filename,
                                    names=[ 'gain', 'offset' ],
                                    delim_whitespace=True,
                                    header=None
            )

            # ahsi channel configuration
            filename = glob.glob( os.path.join( path, f'*_{waveband}IR_Spectralresponse.raw' ) )[ 0 ]
            self.raw = pd.read_csv( filename,
                                    names=[ 'centre', 'width' ],
                                    delim_whitespace=True,
                                    header=None )
            return


    def __init__( self, _args ):

        """
        constructor        
        """

        # call base constructor
        super().__init__( _args )
        self._image = {}
        self._path = None
        self._datetime = None


    def process( self ):

        """
        process ziyuan ahsi datasets
        """

        # load target spectral profile
        target = MatchedFilter.open_target_profile( self._args.target_pathname,
                                                    self._args.active_channels )
        target.insert( 0, 'index_lut', np.arange( len( target ) ) )

        # read ahsi data into numpy data - tranpose for row, channel, column ordering
        img_data = self.open_dataset( self._args.input_pathname )
        img_data = img_data.get_shortwave_radiance( self._args.active_channels )
        img_data = np.transpose( img_data, ( 0, 2, 1 ) )

        # initialise output with nodata
        nodata = -9999 if self.get_shortwave_image().src.nodata is None else self.get_shortwave_image().src.nodata
        out_data = np.zeros( ( img_data.shape[ 0 ], img_data.shape[ -1 ],
                               len( MatchedFilter.OutputBand ) )
        )

        # get target enhancement data
        stats = self.get_target_enhancement( img_data, target, out_data, nodata=nodata )

        # get profile and rcps
        rpcs = self.get_shortwave_image().src.rpcs
        profile = self.get_shortwave_image().src.profile

        # update source profile
        profile.update(
            dtype=rasterio.float32,
            count=len( MatchedFilter.OutputBand ),
            compress='lzw')

        # create directory if not exists
        if not os.path.exists( os.path.dirname( self._args.out_pathname ) ):
            os.makedirs( os.path.dirname( self._args.out_pathname ) )

        # update output data to file
        with rasterio.open( self._args.out_pathname, 'w+', rpcs=rpcs, **profile ) as dst:
            dst.write( np.transpose( out_data, ( 2, 0, 1 ) ) )

        # write stats into csv file via pandas
        df = pd.DataFrame( np.column_stack( ( stats[ 'count' ],
                                              stats[ 'avg' ],
                                              stats[ 'std' ] )
        ), columns=['count','avg','std'] )
        df.to_csv( self._args.out_pathname + '_stats.csv' )

        return


    def open_dataset( self, pathname ):

        """
        load Ziyuan AHSI image, geolocation and ancillary datasets
        """

        # check working with directory
        self._path = pathname
        if not os.path.isdir( pathname ):
            self._path = os.path.dirname( pathname )

        # extract datetime from pathname
        match = re.search( '_[0-9]{8}_', self._path )
        if match is not None:

            # get acquisition datetime
            dt = self._path[ match.span()[ 0 ] : match.span()[ 1 ] ].strip( '_' )
            self._datetime = datetime.strptime( dt, "%Y%m%d" )

            # iterate through optical and shortwave
            for waveband in [ 'VN', 'SW' ]:
                self._image[ waveband ] = Ahsi.Image( self._path, waveband )
        else:
            # unable to parse pathname
            print ( f'AHSI naming convention not found: {self._path}')

        return self


    def get_radiance( self, waveband, channels ):

        """
        Load selected channels of ahsi data into numpy array - use ancillary 
        calibration coefficients for dn to radiance conversion.
        """

        # read dn values from file
        dn = self.get_image( waveband ).src.read( tuple( channels ) )

        # get subset of calibration coefficients
        cal = self._image[ waveband ].cal
        cal = cal [ cal.index.isin( channels - 1 ) ]

        # use cal coefficients to convert dn to radiances
        radiance = np.zeros( dn.shape )
        for idx, row in enumerate( cal.itertuples() ):
            radiance[ idx, :, : ] = dn [ idx, :, : ] * row.gain + row.offset

        return np.transpose( radiance, ( 1, 2, 0 ) )


    def get_shortwave_radiance( self, channels ):
        """
        get shortwave radiance
        """
        return self.get_radiance( 'SW', channels )

    def get_vir_radiance( self, channels ):
        """
        get vir radiance
        """
        return self.get_radiance( 'VN', channels )

    def get_shortwave_image( self ):
        """
        get shortwave image
        """
        return self.get_image( 'SW' )

    def get_vir_image( self ):
        """
        get vir image
        """
        return self.get_image( 'VN' )

    def get_image( self, waveband ):
        """
        get image
        """
        return self._image[ waveband ]


if __name__ == '__main__':

    # parse command line arguments
    args = MatchedFilter.parse_arguments()
    args.active_channels = MatchedFilter.get_active_channels( args )
    args.active_channels = args.active_channels[ 0::args.sampling_rate ]

    # create object and execute processing
    obj = Ahsi( args )
    obj.process()
