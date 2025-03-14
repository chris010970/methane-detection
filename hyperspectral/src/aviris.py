import os
import pandas as pd
import numpy as np
from numpy import dtype

from spectral.io.envi import open as envi_open_file
from spectral.io.envi import create_image as envi_create_image
from spectral.io.envi import dtype_to_envi

from base import MatchedFilter

class Aviris( MatchedFilter ):

    def __init__( self, _args ):

        """
        constructor        
        """

        # call base constructor
        super().__init__( _args )
        self._meta = {}

        # initialise meta items for output
        self._meta[ 'interleave' ] = 'bip'
        self._meta[ 'data type' ] = Aviris.np2_envi_type( np.float64 )

        # get output band names and count
        self._meta[ 'bands' ] = len( MatchedFilter.OutputBand )
        self._meta[ 'band names' ] = [ item.description for item in MatchedFilter.OutputBand ]


    def process( self ):

        """        
        Main control loop
        """

        # load target profile
        target = MatchedFilter.open_target_profile( self._args.target_pathname, self._args.active_channels )
        target.insert( 0, 'index_lut', target.channel_index.values - 1 )

        # load hyperspectral datacube and metadata
        img = Aviris.open_envi_dataset( self._args.input_pathname )
        meta = img.metadata.copy()

        # update metadata for output
        meta.update( self._meta )
        meta[ 'lines' ] = img.nrows

        # remove non output related parameters from metadata
        for kwarg in [ 'smoothing factors', 'wavelength', 'wavelength units', 'fwhm' ]:
            meta.pop( kwarg, None )

        # create new dataset from metadata
        out = Aviris.create_envi_dataset( self._args.out_pathname, meta )

        # retrieve no data - raise exception on +ve values
        nodata = float( meta.get( 'data ignore value', -9999 ) )
        if nodata > 0:
            raise BaseException( f'nodata value={nodata} > 0, values will not be masked' )

        # get target enhancement data
        stats = self.get_target_enhancement(    img._memmap,
                                                target,
                                                out._memmap,
                                                nodata=nodata
        )

        # write stats into csv file via pandas
        df = pd.DataFrame( np.column_stack( ( stats[ 'count' ],
                                              stats[ 'avg' ],
                                              stats[ 'std' ] ) ),
                                              columns=['count','avg','std']
        )
        df.to_csv( self._args.out_pathname + '_stats.csv' )
        return


    @staticmethod
    def np2_envi_type( np_dtype ):

        """
        convert numpy dtype to envi file format data type
        """

        # convert numpy dtype to envi file format data type
        _dtype = dtype(np_dtype).char
        return dtype_to_envi[ _dtype ]


    @staticmethod
    def open_envi_dataset( pathname, interleave='source' ):

        """
        Use PySpectral library to load image file in Envi raster file format. Function returns
        object encapsulating metadata, georeferencing and numpy memory map to data cube of 
        hyperspectral radiances.
        """

        # open file and memory map in read only mode
        img = envi_open_file( pathname + '.hdr', image=pathname )
        _ = img.open_memmap( interleave=interleave, writeable=False )

        return img


    @staticmethod
    def create_envi_dataset( pathname, meta, interleave='source' ):

        """
        Use PySpectral library to create image file in Envi raster file format compliant with 
        metadata passed as functional argument. Function returns object encapsulating metadata, 
        georeferencing and blank numpy memory map with dimensions defined by metadata fields.
        """

        # create output path if not exists
        if not os.path.exists ( os.path.dirname( pathname ) ):
            os.makedirs( os.path.dirname( pathname ) )

        # create envi image object and open memory map in read/write mode
        img = envi_create_image( pathname + '.hdr', meta, force=True, ext='' )
        _ = img.open_memmap( interleave=interleave, writable=True )

        return img


if __name__ == '__main__':

    # parse command line arguments
    args = MatchedFilter.parse_arguments()
    args.active_channels = MatchedFilter.get_active_channels( args )
    args.active_channels = args.active_channels[0::args.sampling_rate]

    # create object and execute processing
    obj = Aviris( args )
    obj.process()
