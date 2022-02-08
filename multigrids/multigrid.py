"""
Multigrid
---------

multigrid.py

This file contains the MultiGrid class

"""

import os
import glob
import copy
from tempfile import mkdtemp

import yaml
import numpy as np
import matplotlib.pyplot as plt
try: 
    import gdal
except ImportError:
    gdal = None

from spicebox import raster, transforms

from .__metadata__ import __version__
from . import figures

from .common import load_or_use_default, GridSizeMismatchError

class MultigridConfigError (Exception):
    """Raised if a multgrid class is missing its configuration"""

class MultigridIOError (Exception):
    """Raised during multigrid IO"""

class MultigridFilterError (Exception):
    """Raised during multigrid Filter ops"""


def open_or_create_memmap_grid(filename, mode, dtype, shape):
    """Initialize or open a memory mapped np.array.
    
    Parameters
    ----------
    filename: str
        Path to file to create or open
    mode: str
        Mode to open file in: 'r', 'r+' or 'w+'.
    dtype: str
        Data type of array. Must be a type suppored by numpy.
    shape: tuple
        Shape of array to create 

    Returns
    -------
    Opened memory mapped array.
    
    """
    if not os.path.exists(filename) and mode in ('r','r+'):
        ## if file dne initialize and delete
        grids = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)           
        del grids
    return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
    

class MultiGrid (object):
    """
        A class to represent a set of multiple related grids of the same 
        dimensions. Implemented usin gnumpy arrays.

    Parameters
    ----------
    *args: list
        List of required arguments, containing exactly 3 items: # rows, 
        # columns, # grids.
        Example call: mg = MultiGrid(rows, cols, n_grids)
    **kwargs: dict
        Dictionary of key word arguments. The following options are valid
        'data_type': data type to use for grid values, defaults 'float'
        'mode': Mode to open memmap file in. Can be 'r+', 'r', 'w+', or 'c'
        'dataset_name': Name of data set, defaults 'Unknown'
        'mask': Mask for the area of interest(AOI) in the MultiGrid.
            Should be a np.array of type bool where true values are
            in the AOI.
            Defaults to None, which generates a mask that sets all 
            grid cells to be part of the AOI
        'grid_names': list of names for each grid in the MultGird, 
            Defaults None.
        'filename': name of file if using memmap as data_model.
            Defaults to None, which will create a temp file.
        'data_model': how to internally represent data, memmap or array.
            Defaults  Memmap.
        'initial_data': Initial data for MultiGrid. Defaults to None. 
    
    Attributes 
    ----------
    config: dict
        'grid_shape': 2 tuple, grid shape (rows, cols)
        'real_shape': 3 tuple, (num_grids, rows, cols)
        'memory_shape': 2 tuple, (num_grids, rows * cols)
        'data_type': data type of grid values
        'mode': Mode memory mapped file is open in. 
        'dataset_name': Name of data set.
        'mask': Mask for the area of interest(AOI) in the MultiGrid grids. 
        'filename': name of memory mapped file.
        'data_model': model of data in memory, 'array', or 'memmap'
        'grid_name_map': map of grid names to gird ids
    grids: np.memmap or np.ndarray 
    """

    def __init__ (self, *args, **kwargs):
        """ Class initializer """
        # print( args )
        # print( kwargs )
        if type(args[0]) is int:
            # print('new')
            init_func = self.new

        if type(args[0]) is str:
            # print('load')
            init_func = self.load

        self.filters = {} 
        self.current_filter = None

        self.config, self.grids = init_func(*args, **kwargs)
        self.config['multigrids_version'] =  __version__



        if "filters" in self.config and type(args[0]) is str:
            if (not self.config['filters'] is None) and \
                    self.config['filters'] != []:
                print('filters', self.config['filters'])
                mg_path,name = os.path.split(args[0])
                name = name[:-4]
                filter_file = os.path.join(mg_path, name + '.filters.data')
                # print(filter_file)
                n = len(self.config['filters'])
                rows, cols = self.config['grid_shape']
                # print((n, rows, cols), self.config['data_type'])
                f_data = np.memmap(
                    filter_file, 
                    mode='r', 
                    dtype=self.config['data_type'],
                    shape = (n, rows, cols)#'z', y, x as it were
                ) 
                # print(f_data)
                self.filters = {}
                for f in self.config['filters']:
                    c = self.config['filters'][f]
                    self.filters[f] = f_data[c].reshape(rows,cols)


        try:
            if type(self.config['raster_metadata']) is dict:
                pass # right-o move along
            else: # old raster metadata fromat
                transform = self.config['raster_metadata'].transform
                projection = self.config['raster_metadata'].projection
                nX = self.config['raster_metadata'].nX
                nY = self.config['raster_metadata'].nY


                self.config['raster_metadata'] = { 
                    'transform': transform,
                    'projection': projection,
                    'x_size': nX,
                    'y_size': nY,
                }
        except KeyError:
            pass # no raster meta is a ok


    def __del__ (self):
        """deconstructor for class"""
        if hasattr(self, 'config'):

            del(self.config)
        if hasattr(self, 'grids'):
            del(self.grids)

        if hasattr(self,' _tempfile_loc'):
            os.remove(self._tempfile_loc)

            

    # def __getattr__(self, attr):
    #     """get attribute, allows access to config dictionary values
    #     as class attributes 

    #     Paramaters
    #     ----------
    #     attr: str
    #         attribute. spaces in this paramater are replaced with '_' if 
    #         the space version of the attribute is not found.

    #     Raises
    #     ------
    #     AttributeError:
    #         if Attribute is not found.

    #     Returns
    #     -------
    #     value of attribute
    #     """
    #     # if not hasattr(self, 'config'):
    #     #     raise MultigridConfigError( "config dictionary not found" )
    #     # try:
    #     if attr == 'config':
    #         return self.config
    #     elif attr in self.config and attr != 'config'  and attr != 'config':
    #         return self.config[attr]
    #     elif attr.replace('_',' ') in self.config and attr != 'config':
    #         return self.config[attr.replace('_',' ')]
    #     else:
    #         s = "'" + self.__class__.__name__ + \
    #             "' object has no attribute '" + attr + "'"
    #         raise AttributeError(s)
    #     # except(AttributeError) as e:
    #     #     return 'not attr'
        
    def __repr__ (self):
        """Get string representation of object
        
        Returns
        -------
        string
        """
        # print 'something'
        try: 
            return str(self.grids.reshape(self.config['real_shape']))
        except AttributeError:
            return "object not initialized"
        
    def __getitem__(self, key): 
        """Get item function
        
        Parameters
        ----------
        key: str or int
            A grid named in the grid_names list, and grid_name_map, or
        A int index for one of the grids.

        Returns
        -------
        np.array like
            Grid from the multigrid with the shape self.config['grid_shape']
        
        """
        if type(key) in (str,):
            key = self.get_grid_number(key)

        _filter = self.filters[self.current_filter] if self.current_filter else 1
        
        return self.grids.reshape(self.config['real_shape'])[key].reshape(self.config['grid_shape']) * _filter

    def __setitem__(self, key, value):
        """Set item function
        
        Parameters
        ----------
        key: str or int
            A grid named in the grid_names list, and grid_name_map, or
        A int index for one of the grids.
        value: np.array like
            Grid that is set. Should have shape of self.config['grid_shape'].
        """
        if type(key) in (str,):
            key = self.get_grid_number(key)
        # if type(key) in (tuple, int, slice):
        self.grids[key] = value.flatten()
    
    def __eq__(self, other):
        """Equality Operator for MultiGrid object 

        Returns
        -------
        Bool:
            True if intreal grids are equal
        """
        if other is self:
            return True
        return (self.grids == other.grids).all()

    def new(self, *args, **kwargs):
        """Does setup for a new MultGrid object
        
        Parameters
        ----------
        *args: list
            Variable length list of arguments. Needs 3 itmes.
            where the first is the number of rows in each grid, the
            second is the number of columns, and the third is the 
            number of grids in the MultiGrid. All other values are
            ignored. 
        **kwargs: dict
            Dictionary of key word arguments. The following options are valid
            'data_type': data type to use for grid values, defaults 'float'
            'mode': Mode to open memmap file in. Can be 'r+', 'r', 'w+', or 'c'
            'dataset_name': Name of data set, defaults 'Unknown'
            'mask': Mask for the area of interest(AOI) in the MultiGrid.
                Should be a np.array of type bool where true values are
                in the AOI.
                Defaults to None, which generates a mask that sets all 
                grid cells to be part of the AOI
            'grid_names': list of names for each grid in the MultGird, 
                Defaults None.
            'filename': name of file if using memmap as data_model.
                Defaults to None, which will create a temp file.
            'data_model': how to internally represent data, memmap or array.
                Defaults  Memmap.
            'initial_data': Initial data for MultiGrid. Defaults to None. 

        Returns
        -------
        Config: dict
            dict to be used as config attribute.
        Grids: np.array like
            array to be used as internal memory.
        """

        config = {}
        config['grid_shape']= (args[0], args[1])
        config['num_grids'] = args[2]
        config['memory_shape'] = self.get_memory_shape(config)
        config['real_shape'] = self.get_real_shape(config)
        config['data_type'] = load_or_use_default(
            kwargs, 'data_type', 'float32'
        )
        config['mode'] = load_or_use_default(kwargs, 'mode', 'r+')
        config['dataset_name'] = load_or_use_default(
            kwargs, 'dataset_name', 'Unknown'
        )

        config['mask'] = load_or_use_default(kwargs, 'mask', None)

        if config['mask'] is None:
            config['mask'] = \
                np.ones(config['grid_shape']) == \
                np.ones(config['grid_shape'])

        grid_names = load_or_use_default(kwargs, 'grid_names', [])
        
        if len(grid_names) > 0 and config['num_grids'] != len(grid_names):
            raise GridSizeMismatchError( 'grid name size mismatch' )
        config['grid_name_map'] = self.create_name_map(grid_names)
            

        config['filename'] = load_or_use_default(kwargs, 'filename', None)
        config['data_model'] = load_or_use_default(
            kwargs, 'data_model', 'memmap'
        )
        if config['data_model'] != 'memmap':
            config['data_model'] = 'array'

        grids = self.setup_internal_memory(config)
        
        init_data = load_or_use_default(kwargs, 'initial_data', None)
        if not init_data is None:
            grids = init_data.reshape(config['memory_shape'])

        config['filters'] = None
        
        return config, grids 

    def load(self, file):
        """Load a MultiGird object from yaml metadata file and data file
        
        Parameters
        ----------
        file: path
            Path to the metadata file for the MultiGrid object.

        Returns
        -------
        Config: dict
            dict to be used as config attribute.
        Grids: np.array like
            array to be used as internal memory.
        """
        with open(file) as conf_text:
            config = yaml.load(conf_text, Loader=yaml.Loader)
        config['cfg_path'] = os.path.split(file)[0]
        config['memory_shape'] = self.get_memory_shape(config)
        config['real_shape'] = self.get_real_shape(config)
        grids = self.setup_internal_memory(config)
        return config, grids

    def save(self, file=None, grid_file_ext = '.mgdata'):
        """Save MiltiGrid Object to metadata file and data file
        The metadata file cantinas the config info, and the data file
        contains the grids. 

        Parameters
        ----------
        file: path
            Path to the metadata file (yaml) to save. The data file name will
            be genetated from metadata name if a data file does not all ready
            exist. 
        grid_file_ext: str, defaults: '.mgdata' 
            The extension to save the data file as if the datafile does not
            already exist. The data is saved as a memory mapped Numpy array,
            so the extension is more for description than any thing else.
        """
        if file is None and self.config['dataset_name'].lower() == "Unknown":
            msg = "Error saving multigrid, no filename. Provided" +\
                " a filename to the save function, or set the dataset_name in"+\
                " the multigrid config to a value other than Unknown or" +\
                " unknown."
            raise MultigridIOError()
        elif file is None and self.config['dataset_name'].lower() != "Unknown":
            file = self.config['dataset_name'].lower().replace(' ','_') + '.yml'

        s_config = copy.deepcopy(self.config)
        if s_config['data_model'] is 'array' or s_config['filename'] is None:
            try:
                path, grid_file = os.path.split(file)
            except ValueError:
                path, grid_file = './', file
            if grid_file[0] == '.':
                grid_file = '.' + grid_file[1:].split('.')[0] + grid_file_ext
            else:
                grid_file = grid_file.split('.')[0] + grid_file_ext
            data_file = os.path.join(path,grid_file) 
            save_file = np.memmap(
                data_file, 
                mode = 'w+', 
                dtype = s_config['data_type'], 
                shape = self.grids.shape 
            )
            save_file[:] = self.grids[:]
            del save_file ## close file
            s_config['filename'] = os.path.split(data_file)[1]
        
        del s_config['memory_shape']
        del s_config['real_shape']

        s_config['mode'] = 'r+'

        if self.filters != {}:
            try:
                path, filter_file = os.path.split(file)
            except ValueError:
                path, filter_file = './', file

            if filter_file[0] == '.':
                filter_file = '.' + filter_file[1:].split('.')[0] + '.filters.data'
            else:
                filter_file = filter_file.split('.')[0] + '.filters.data'
            filter_file = os.path.join(path,filter_file)

            n = len(self.filters)
            rows, cols = self.config['grid_shape']
            f_data = np.memmap(
                filter_file, 
                mode='w+', 
                dtype=s_config['data_type'],
                shape = (n, rows, cols)#'z', y, x as it were
            ) 

            # stack filters and create map to index location
            f_map = {}
            c = 0
            for f in self.filters:
                f_data[c][:] = self.filters[f][:]
                f_map[f] = c

            del f_data ## close file
            
            s_config['filters'] = f_map

        with open(file, 'w') as sfile:
            sfile.write('#Saved ' + self.__class__.__name__ + " metadata\n")
            yaml.dump(s_config, sfile, default_flow_style=False)
        
    def create_name_map(self, grid_names):
        """Creates a dictionary to map string grid names to their 
        interger index values. Used to initialize gird_name_map
        
        Paramaters
        ----------
        grid_names: list of strings
            List of grid names. Length == num_grids

        Returns
        -------
        Dict:
            String: int, key value pairs
        """
        return {grid_names[i]: i for i in range(len(grid_names))}   

    def add_filter(self, name, data, force=False):
        """add a filter to the data set

        Parameters
        ----------
        name: str
            name used to set/ access fileter
        data: np.array 
            filter of shape config['grid_shape'] is used as multiplier 
            by getter functions with filters is set via set_filter() call
        force: bool,  default False
            if True existing fliters named `name` are overwritten
        
        """

        if name in self.filters and force == False:
            raise MultigridFilterError("filters contains %s filter" % name)

        if data.shape != self.config['grid_shape']:
            raise MultigridFilterError("filter shape does not match grid shape")

    
        self.filters[name] = data

    def set_filter(self, name):
        """
        """
        if name in self.filters or name is None:
            self.current_filter = name
        else:
            raise MultigridFilterError("invalid filter: %s" % name)


    def setup_internal_memory(self, config):
        """Setup the internal memory representation of grids

        Parameters
        ----------
        config: dict
            Should have keys 'filename', 'data_model', 'data_type', 
        and 'memory_shape'. 
            'filename': name of file to write or None
            'data_model': Model for data representation: 'array' or 'memmap'
            'data_type': 
                String type of data. Must be type supported by np.arrays.
            'memory_shape': Tuple
                shape of grids as represented in memory 

        Returns
        -------
        grids: np.array or np.memmap
        """
        filename = config['filename']
        if config['filename'] is None and config['data_model'] == 'memmap':
            # print "a"
            filename = os.path.join(mkdtemp(), 'temp.dat')
            self._tempfile_loc = filename
        elif not config['filename'] is None and not os.path.exists(filename):
            # print "b", filename
            filename = os.path.split(filename)[1]
            filename = os.path.join(config['cfg_path'], filename)
            
        if config['data_model'] == 'memmap':
           
            
            # print filename
            grids = open_or_create_memmap_grid(
                filename, 
                config['mode'], 
                config['data_type'], 
                config['memory_shape']
            )
        else: # array
            grids = np.zeros(config['memory_shape'])
        return grids 
    
    def get_memory_shape (self, config):
        """Construct the shape needed for multigrid in memory from 
        configuration. 

        Parameters
        ----------
        config: dict
            Must have keys 'num_grids' an int, 'grid_shape' a tuple of 2 ints

        Returns
        -------
        Tuple
            (num_grids, flattened shape of each grid )
        """
        return (config['num_grids'], 
            config['grid_shape'][0] * config['grid_shape'][1])

    def get_real_shape (self, config):
        """Construct the shape that represents the real shape of the 
        data for the MultiGird.

        Parameters
        ----------
        config: dict
            Must have keys 'num_grids' an int, 'grid_shape' a tuple of 2 ints

        Returns
        -------
        Tuple
            ('num_grids', 'rows', 'cols')
        """
        return (config['num_grids'], 
            config['grid_shape'][0], config['grid_shape'][1])

    def get_grid_number(self, grid_id):
        """Get the Grid number for a grid id
        
        Parameters
        ----------
        grid_id: int or str
            if an int, it should be the grid number.
            if a str, it should be a grid name.

        Returns
        -------
        int
            gird id
        """
        return grid_id if type(grid_id) is int else self.config['grid_name_map'][grid_id]
    
    def get_grid(self, grid_id, flat = True):
        """Get a grid
        
        Parameters
        ----------
        grid_id: int or str
            if an int, it should be the grid number.
            if a str, it should be a grid name.
        flat: bool, defaults true
            returns the grid as a flattened array.

        Returns
        -------
        np.array
            1d if flat, 2d otherwise.
        """
        if flat:
            return self[grid_id].flatten()
        return self[grid_id]

    def set_grid(self, grid_id, new_grid):
        """Set a grid
         Parameters
        ----------
        grid_id: int or str
            if an int, it should be the grid number.
            if a str, it should be a grid name.
        new_grid: np.array like
            Grid to set. must be able to reshape to grid_shape.
        """
        self[grid_id] = new_grid.reshape(self.config['grid_shape'])

    def save_figure(
            self, grid_id, filename, figure_func=figures.default, figure_args={}, data=None
        ):
        """
        """
        if data is None:
            data = self[grid_id].astype(float)
        # data[np.logical_not(self.mask)] = np.nan
        
        if not 'title' in figure_args:
            figure_args['title'] = self.config["dataset_name"] 
            if not grid_id is None:
                figure_args['title' ]+= ' ' + str( grid_id )
        fig = figure_func(data, figure_args)
        plt.savefig(filename)
        plt.close()

    def show_figure(self, grid_id, figure_func=figures.default, figure_args={}, data=None):
        """
        """
        if data is None:
            data = self[grid_id].astype(float)
        # data[np.logical_not(self.mask)] = np.nan
        if not 'title' in figure_args:
            figure_args['title'] = self.config["dataset_name"] 
            if not grid_id is None:
                figure_args['title' ] += ' ' + str( grid_id )
        fig = figure_func(data, figure_args)
        plt.show()
        plt.close()

        
    def save_all_figures(
            self, dirname, figure_func=figures.default, figure_args={}, extension='.png'
        ):
        """
        """
        grids = self.config['grid_name_map']
        if grids == {}:
            grids = range(self.num_grids)

        for grid in grids:
            filename = os.path.join(
                dirname, 
                (self.config["dataset_name"]  + '_' + str(grid) + extension).replace(' ','_')
            )
            figure_args['title'] = self.config["dataset_name"].replace('-', ' ')  + ' ' + str(grid)
            self.save_figure(grid, filename, figure_func, figure_args)


    def save_as_geotiff(self, filename, grid_id, **kwargs):
        """save a grid as a tiff file
        """
        if gdal is None:
            raise IOError("gdal not found: cannot save tif")
        

        try:
            if type(self.config['raster_metadata']) is dict:
                transform = self.config['raster_metadata']['transform']
                projection = self.config['raster_metadata']['projection']
        #     else: # old raster metadata fromat
        #         transform = self.config['raster_metadata'].transform
        #         projection = self.config['raster_metadata'].projection
        except KeyError:
            raise IOError("No Raster Metadata Found: cannot save tiff")
        datatype = gdal.GDT_Float32

        # print (projection)
        
        # print (transform)
        data = self[grid_id].astype(np.float32)

        raster.save_raster(filename, data, transform, projection, datatype)



    def save_all_as_geotiff(self, dirname, **kwargs):
        """save all grid as a tiff file
        """
        grids = self.config['grid_name_map']
        if grids == {}:
            grids = range(self.config['num_timesteps'])

        try:
            if self.config['start_timestep'] != 0:
                grids = range(
                    self.config['start_timestep'], 
                    self.config['start_timestep']+self.config['num_timesteps']
                )
        except KeyError:
            pass

        if 'base_filename' in kwargs:
            bfn = kwargs['base_filename']
        else:        
            bfn = self.config["dataset_name"] 
        
        for grid in grids:
            filename = os.path.join(
                dirname, 
                (bfn + '_' + str(grid) + '.tif').replace(' ','_')
            )
            self.save_as_geotiff(filename, grid, **kwargs)

    def get_range(self):
        """get the range of time steps"""
        return range(
            self.config['start_timestep'], 
            self.config['start_timestep'] + self.config['num_timesteps']
        )

    def get_as_ml_features(self, grid, mask = None, train_range=None ):
        """Get the data in a way that can be used in ML methods
        """
        features = []
        if mask is None:
            mask = np.ones(self.config['grid_shape'])
            mask = mask == mask

        if train_range is None:
            train_range = self.get_range()

        for ts in train_range:
            if grid is None:
                temp = np.array(self[ts])
            else:
                temp = np.array(self[grid, ts])
            temp[np.logical_not(mask)] = np.nan
            features += list(temp[mask])
        return np.array(features)

    def clip_grids(self, extent, location_format="ROWCOL", verbose=False):
        """Clip the desired extent from the multigrid. Returns a new 
        Multigrid with the smaller extent.

        parameters
        ----------
        extent: tuple
            4 tuple containing top left and bottom right coordinates to clip
            data to


            (row_tr, col_tr, row_bl, col_bl) if location format == "ROWCOL"
            (east_tr, north_tr, east_bl, north_bl) if location format == "GEO"
            (Long_tr, Lat_tr, Long_bl, Lat_bl) if location == "WGS84"
        radius: Int, default 50
            number of pixels around center to include in zoom
        location_format: str, default "ROWCOL"
            "ROWCOl", "GEO", or "WGS84" to indcate if locations are in
            pixel, map, or WGS84 format
        verbose: bool, default False
    
        returns
        -------
        multigrid.Multigrid
        """
        top_l = extent[0], extent[1]
        bottom_r = extent[2], extent[3]

        transform = self.config['raster_metadata']['transform']
        projection = self.config['raster_metadata']['projection']

        if location_format == "WGS84":
            top_l = transforms.from_wgs84(top_l, projection)
            top_l = transforms.to_pixel(top_l, transform).astype(int)

            bottom_r = transforms.from_wgs84(bottom_r, projection)
            bottom_r = transforms.to_pixel(bottom_r, transform).astype(int)
        elif location_format == "GEO":
            top_l = transforms.to_pixel(top_l, transform).astype(int)
            bottom_r = transforms.to_pixel(bottom_r, transform).astype(int)

        if verbose:
            print ('top left', top_l)
            print ('bottom right', bottom_r)

        data = self.grids[0].reshape(self.config['grid_shape'])
        
        view = raster.zoom_box(data, top_l, bottom_r)
        n_grids = self.config['num_grids']
        rows, cols = view.shape
        
        view = type(self)(
            rows, cols, n_grids,
            data_type=self.config['data_type'],
        )
        view.config['grid_name_map'] = self.config['grid_name_map']

        try:
            view.config['description'] = \
                self.config['description'] + ' clipped to' + str(extent)
        except KeyError:
             view.config['description'] = + 'Unknown clipped to' + str(extent)
        
        try:
            view.config['dataset_name'] = \
                self.config['dataset_name'] + ' clipped to' + str(extent)
        except KeyError:
            view.config['dataset_name'] = 'Unknown clipped to' + str(extent)

        raster_meta = self.config['raster_metadata']
        view_transform = raster.get_zoom_box_geotransform(
            raster_meta, top_l, bottom_r
        )



        for idx in range(len(self.grids)):
            grid = self.grids[idx].reshape(self.config['grid_shape'])
            zoom = raster.zoom_box(grid, top_l, bottom_r)
            view.grids[idx][:] = zoom.flatten()

        view.config['mask'] = raster.zoom_box(
            self.config['mask'], top_l, bottom_r
        )

        view.config['raster_metadata'] = copy.deepcopy(raster_meta)
        view.config['raster_metadata']['transform'] = view_transform

        for filter in self.config['filters']:
            filter_data = self.filters[filter]
            filter_data = raster.zoom_box(filter_data, top_l, bottom_r)
            view.add_filter(filter, filter_data )


        return view

    def zoom_to(
            self, location, radius=50, location_format="ROWCOL", verbose=False
        ):
        """zoom in to center location

        parameters
        ----------
        location: tuple
            (row, col) if location format == "ROWCOL"
            (east, north) if location format == "GEO"
            (Long, Lat) if location == "WGS84"
        radius: Int, default 50
            number of pixels around center to include in zoom
        location_format: str, default "ROWCOL"
            "ROWCOl", "GEO", or "WGS84" to indcate if location is in
            pixel, map, or WGS84 format
        verbose: bool, default False
    
        returns
        -------
        multigrid.Multigrid
        """
        location = np.array(location).reshape((2,))
        loc_orig = copy.deepcopy(location)

        transform = self.config['raster_metadata']['transform']
        projection = self.config['raster_metadata']['projection']

        if location_format == "WGS84":
            location = transforms.from_wgs84(location, projection)
            location = transforms.to_pixel(location, transform).astype(int)
        elif location_format == "GEO":
            location = transforms.to_pixel(location, transform).astype(int)
        # else:  # ROWCOL
        #     pass

        if verbose:
            print(location)


        data = self.grids[0].reshape(self.config['grid_shape'])
        
        view = raster.zoom_to(data, location, radius)
        n_grids = self.config['num_grids']
        rows, cols = view.shape
        
        view = type(self)(
            rows, cols, n_grids,
            data_type=self.config['data_type'],
        )
        view.config['grid_name_map'] = self.config['grid_name_map']

        try:
            view.config['description'] = \
                self.config['description'] + ' zoom to ' + str(loc_orig)
        except KeyError:
             view.config['description'] = 'Unknown zoom to ' + str(loc_orig)
        
        try:
            view.config['dataset_name'] = \
                self.config['dataset_name'] + ' zoom to ' + str(loc_orig)
        except KeyError:
            view.config['dataset_name'] = 'Unknown zoom to ' + str(loc_orig)

        raster_meta = self.config['raster_metadata']
        view_transform = raster.get_zoom_geotransform(
            raster_meta, location, radius
        )



        for idx in range(len(self.grids)):
            grid = self.grids[idx].reshape(self.config['grid_shape'])
            zoom = raster.zoom_to(grid, location, radius)
            view.grids[idx][:] = zoom.flatten()

        view.config['mask'] = raster.zoom_to(
            self.config['mask'], location, radius
        )

        view.config['raster_metadata'] = copy.deepcopy(raster_meta)
        view.config['raster_metadata']['transform'] = view_transform

        for filter in self.config['filters']:
            filter_data = self.filters[filter]
            filter_data = raster.zoom_to(filter_data, location, radius)
            view.add_filter(filter, filter_data )


        return view

    def at(self, location, radius=0, location_format="ROWCOL", verbose=False):
        """
        """
        location = np.array(location).reshape((2,))
        loc_orig = copy.deepcopy(location)

        transform = self.config['raster_metadata']['transform']
        projection = self.config['raster_metadata']['projection']

        if location_format == "WGS84":
            location = transforms.from_wgs84(location, projection)
            location = transforms.to_pixel(location, transform).astype(int)
        elif location_format == "GEO":
            location = transforms.to_pixel(location, transform).astype(int)
        # else:  # ROWCOL
        #     pass

        if verbose:
            print(location)

        zoom_ts = []

        for idx in range(len(self.grids)):
            ## TODO imporove speed with better indexing
            grid = self.grids[idx].reshape(self.config['grid_shape'])
            zoom = raster.zoom_to(grid, location, radius)
            zoom_ts.append(zoom)
        return np.array(zoom_ts)

    def get_max_locations(self, location_format="ROWCOL", verbose=False, top_n = 3, start_at=0):

        max_map = {}
        transform = self.config['raster_metadata']['transform']
        projection = self.config['raster_metadata']['projection']

        
        for idx in range(start_at,len(self.grids)):
            if verbose:
                print(idx)
            grid = self.grids[idx].reshape(self.config['grid_shape'])

            top = np.unique(grid[~np.isnan(grid)])[-1 * top_n:]

            max_map[idx] = {}

            count = top_n
            if verbose:
                print(top)
            for val in top:
                if verbose:
                    print(val)
                locations = np.array(np.where(grid==val)).T

                new_locs = []
                if location_format != "ROWCOL":
                    for lix in range(len(locations)): 
                           
                        if verbose:
                            print("geo correcting")
                        location = locations[lix] 
                        loc_i = locations[lix] 
                        if location_format == "WGS84":
                            location = transforms.to_geo(location, transform)
                            location = transforms.to_wgs84(location, projection)
                        elif location_format == "GEO":
                            location = transforms.to_geo(location, transform)
                        new_locs.append(list(location)) 
                        print(lix, loc_i, location) 
                else: 
                    new_locs = locations

                max_map[idx][count] = {"value": val, "at": new_locs}
                count -= 1
            
        return max_map
        
 

    def get_grids_at_keys(self, keys):
        """return the grids for the given keys

        Parameters
        ----------
        keys: list
            list of grids
        
        Returns
        -------
        np.array
        """
        select = np.zeros([ 
            len(keys), 
            self.config['grid_shape'][0],
            self.config['grid_shape'][1]
        ])
        c = 0
        for k in keys:
            select[c] = self[k]
            c += 1
        return select

    def calc_statistics_for (self, keys, stat_fucn=np.mean, axis = 0):
        """Calculate the statstics for a given substet of grids

        Parameters
        ----------
        keys: list
            list of keys into the multigrid
        """
        data = self.get_grids_at_keys(keys)
        return stat_fucn(data, axis)

    def clone(self):
        """
        """
        new = copy.deepcopy(self)
        
        tmp = mkdtemp()
        filename = os.path.join(tmp, 'temp.dat')

        grids = np.memmap(
            filename, 
            dtype=new.config['data_type'], 
            mode='w+', 
            shape=new.config['memory_shape']
        )           
        del grids
       
        new.grids = np.memmap(
            filename, 
            dtype=new.config['data_type'], 
            mode='r+', 
            shape=new.config['memory_shape']
        )

        new.grids[:] = self.grids[:]

        new.config['filename'] = None
        new.config['dataset_name'] = 'Clone-of-' + new.config['dataset_name']

        
        return new

    def apply_function (self, func):
        """
        """
        new = self.clone()
        for grid in range(len(new.grids)): 
            # print(grid) 
            # print(new.grids[grid])
            new.grids[grid][:] = func(self.grids[grid])
        return new

    def create_subset(self, subset_grids):
        """creates a multigrid containting only the subset_girds

        parameters
        ----------
        subset_grids: list
        """
        rows = self.config['grid_shape'][0]
        cols = self.config['grid_shape'][1]
        n_grids = len(subset_grids)
        subset = type(self)(rows, cols, n_grids, 
            grid_names = subset_grids,
            data_type=self.config['data_type'],
            mask = self.config['mask'],
            
        )

        try:
            subset.config['description'] = \
                self.config['description'] + ' Subset.'
        except KeyError:
            subset.config['description'] = 'Unknown subset.'
        
        try:
            subset.config['dataset_name'] = \
                self.config['dataset_name'] + ' Subset.'
        except KeyError:
            subset.config['dataset_name'] = 'Unknown subset.'

        ngnm = {}
        for idx, grid in enumerate(subset_grids):
            # print(self[grid][:])
            # print(idx)
            subset[idx] = self[grid][:]
            ngnm[grid] = int(idx)



        subset.filters = self.filters
        subset.config['filters'] = self.config['filters']
        subset.config['grid_name_map'] = ngnm 

        subset.config['raster_metadata'] = self.config['raster_metadata'] 

        return subset
        

def create_example():
    """create and return an example MultiGrid

    Returns
    -------
    MultiGrid
    """
    g_names = ['first', 'second']

    init_data = np.ones([2,5,10])
    init_data[1] += 1

    t1 = MultiGrid(5, 10, 2, grid_names = g_names, initial_data = init_data)
    
    return t1

