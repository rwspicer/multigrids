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
from datetime import datetime, date

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

# from .common import common.load_or_use_default, GridSizeMismatchError
from . import common, errors

def is_subgrids_key(key):
    """Test if key is subgrids key

    Parameters
    ----------
    key: any

    Returns
    -------
    Bool
    """
    return type(key) is tuple and type(key[0]) in (list, range, slice)

def is_grids_key(key):
    """Test if key is grids key

    Parameters
    ----------
    key: any

    Returns
    -------
    Bool
    """
    return not is_subgrids_key(key) and \
            type(key) in [range, slice] or \
            type(key) is tuple and \
            not type(key[1]) in (np.ndarray, list, range, slice)

def is_subgrid_key(key):
    """Test if key is subgrid key

    Parameters
    ----------
    key: any

    Returns
    -------
    Bool
    """
    return not is_grids_key(key) and \
            type(key) is tuple and len(key) > 1

def is_grid_key(key):
    """Test if key is grid key

    Parameters
    ----------
    key: any

    Returns
    -------
    Bool
    """
    return not is_subgrid_key(key) and type(key) in (int, str,  datetime, date)


class MultiGrid (object):
    """
    A class to represent a set of multiple related grids of the same 
    dimensions. Implemented using memory mapped numpy arrays.

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
    _is_temp: bool
        attribute to indicate if internal memmap is a tempfile
    filters: dict
        dictionary of flites that may be applied to data when accessing
    
    """
    def __init__ (self, *args, **kwargs):
        """ Class initializer """
        # print('init')
        self._is_temp = False
        self.mask_file = None
        self.filter_file = None

        if type(args[0]) is int:
            init_func = self.new

        if type(args[0]) is str:
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
                self.filter_file = filter_file
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


        if 'raster_metadata' in self.config:
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

        if not 'grid_name_map' in self.config:
            self.configure_grid_name_map(kwargs)
        
        if 'save_to' in kwargs and not kwargs['save_to'] is None:
            self.save(kwargs['save_to'])

        # except KeyError:
        #     pass # no raster meta is a ok

    def __del__ (self):
        """deconstructor for class"""
        if hasattr(self, 'config'):
            del(self.config)
        
        try:
            loc = self.grids.filename
        except AttributeError:
            loc = None

        if hasattr(self, 'grids'):
            del(self.grids)

        if self._is_temp and not loc is None and os.path.exists(loc):
            os.remove(loc)

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
        if is_subgrids_key(key):
            ## multiple subgrids
            nk = key[0]
            # if type(nk) is slice:
            #     nk = self.lookup_grid_range(
            #         nk.start,nk.stop,(nk.step if nk.step else 1)
            #     )
            grids = self.get_subgrids(nk, key[1:], False)

        elif is_grids_key(key):
            ## multiple Grids
            # if type(key) is slice:
            #     key = self.lookup_grid_range(
            #         key.start,key.stop,(key.step if key.step else 1)
            #     )
            grids = self.get_grids(key, False) 

        elif is_subgrid_key(key): ## single subgrid
            grids = self.get_subgrid(key[0], key[1:], False)   

        elif is_grid_key(key):
            grids = self.get_grid(key, False)
        else:
            raise KeyError('Key type is not recognized:', key)

        return grids 

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
        if is_subgrids_key(key):
            ## multiple subgrids
            nk = key[0]
            # if type(nk) is slice:
            #     nk = range(nk.start,nk.stop,(nk.step if nk.step else 1))
            self.set_subgrids(nk, key[1:], value)

        elif is_grids_key(key):
            ## multiple Grids
            if type(key) is slice:
                key = range(key.start,key.stop,(key.step if key.step else 1))
            self.set_grids(key, value) 

        elif is_subgrid_key(key): ## single subgrid
            self.set_subgrid(key[0], key[1:], value)   

        elif is_grid_key(key):
            self.set_grid(key, value)
        else:
            raise KeyError('Key type is not recognized:', key)

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
            Variable length list of arguments. Needs 3 items.
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
            'grid_names': list of names for each grid in the MultiGird, 
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
        config['memory_shape'] = self.create_memory_shape(config)
        config['real_shape'] = self.create_real_shape(config)
        config['data_type'] = common.load_or_use_default(
            kwargs, 'data_type', 'float32'
        )
        config['mode'] = common.load_or_use_default(kwargs, 'mode', 'r+')
        config['dataset_name'] = common.load_or_use_default(
            kwargs, 'dataset_name', 'Unknown'
        )

        config['mask'] = common.load_or_use_default(kwargs, 'mask', None)

        if config['mask'] is None:
            config['mask'] = \
                np.ones(config['grid_shape']) == \
                np.ones(config['grid_shape'])

        # grid_names = common.load_or_use_default(kwargs, 'grid_names', [])
        # config['grid_name_map'] = self.create_grid_name_map(grid_names, config)
            

        config['filename'] = common.load_or_use_default(
            kwargs, 'filename', None
        )
        config['data_model'] = common.load_or_use_default(
            kwargs, 'data_model', 'memmap'
        )
        if config['data_model'] != 'memmap':
            config['data_model'] = 'array'

        init_data = common.load_or_use_default(kwargs, 'initial_data', None)

        
        if not init_data is None and type(init_data) is np.memmap:
            # plt.imshow(init_data[0])
            # plt.show()
            grids = init_data.reshape(config['memory_shape'])
            config['filename'] = grids.filename
        elif  not init_data is None:
            grids = self.setup_internal_memory(config)
            grids[:] = init_data.reshape(config['memory_shape'])
        else:
            grids = self.setup_internal_memory(config)
        
        # print(grids, type(grids), grids.filename)
        
        

        
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
        cfg_path = os.path.split(file)[0]

        if os.path.split(config['filename'])[0] != cfg_path:
            config['filename'] = os.path.join(cfg_path, config['filename'])
        if type(config['mask']) is str and \
                os.path.split(config['mask'])[0] != cfg_path:
            config['mask'] = os.path.join(cfg_path, config['mask'])  

        if 'filters' in config:
            if type(config['filters']) is str and \
                    os.path.split(config['filters'])[0] != cfg_path:
                config['filters'] = os.path.join(cfg_path, config['filters'])  
        else:
            config['filters'] = None
        
        

        config['memory_shape'] = self.create_memory_shape(config)
        config['real_shape'] = self.create_real_shape(config)
        grids = self.setup_internal_memory(config)

        if type(config['mask']) is str and os.path.isfile(config['mask']):
            self.mask_file = config['mask']
            config['mask'] = np.load(config['mask'])
            

        return config, grids

    def save(self, file=None, grid_file_ext = '.grids.data'):
        """Save MultiGrid Object to metadata file and data file
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
            raise errors.MultigridIOError()
        elif file is None and self.config['dataset_name'].lower() != "Unknown":
            file = self.config['dataset_name'].lower().replace(' ','_') + '.yml'

        s_config = copy.deepcopy(self.config)

        try:
            path, grid_file = os.path.split(file)
        except ValueError:
            path, grid_file = './', file

        if grid_file[0] == '.':
            grid_file = '.' + grid_file[1:].split('.')[0] + grid_file_ext
        else:
            grid_file = grid_file.split('.')[0] + grid_file_ext
        data_file = os.path.join(path,grid_file) 
        # print(s_config['filename'], self._is_temp)
        if s_config['data_model'] == 'array':
            
            
            save_file = np.memmap(
                data_file, 
                mode = 'w+', 
                dtype = s_config['data_type'], 
                shape = self.grids.shape 
            )
            save_file[:] = self.grids[:]
            del save_file ## close file
            s_config['filename'] = os.path.split(data_file)[1]

        if s_config['filename'] is None or self._is_temp:
            current = self.grids.filename
            # print(current, self.grids.filename, s_config['filename'])
            t_shape =  self.grids.shape 
            del(self.grids)
            os.rename(current, data_file)
            # from time import sleep
            # sleep(10)
            self.grids =  np.memmap(
                data_file, 
                mode = 'w+', 
                dtype = s_config['data_type'], 
                shape = t_shape
            )
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
                filter_file = \
                    '.' + filter_file[1:].split('.')[0] + '.filters.data'
            else:
                filter_file = filter_file.split('.')[0] + '.filters.data'
            filter_file = os.path.join(path,filter_file)
            self.filter_file = filter_file 
            

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
        else:
            s_config['filters'] = None

        if 'mask' in s_config:
            try:
                path, mask_file = os.path.split(file)
            except ValueError:
                path, mask_file = './', file
            
            if mask_file[0] == '.':
                mask_file = '.' + mask_file[1:].split('.')[0] + '.mask.data'
            else:
                mask_file = mask_file.split('.')[0] + '.mask.data'
            mask_file = os.path.join(path, mask_file)
            self.mask_file = mask_file

            np.save(mask_file, self.config['mask'])
            os.rename(mask_file + '.npy', mask_file)
            s_config['mask'] = os.path.split(mask_file)[-1]

            
        
        ### ensure filename is not an absolute path in saved yml metadata
        filename = s_config['filename']
        while os.path.split(filename)[0] != "":
            filename = os.path.split(filename)[1]
        s_config['filename'] = filename

        with open(file, 'w') as sfile:
            sfile.write('#Saved ' + self.__class__.__name__ + " metadata\n")
            yaml.dump(s_config, sfile, default_flow_style=False)

        

        ## if were saving a memmap make sure the new mg object is pointing
        ## to the right file
        # print(s_config)
        # if hasattr(self.grids, 'filename') and \
        #         self.grids.filename != s_config['filename'] and \
        #         self._is_temp:
        #     # print(self.grids.filename, s_config['filename'])
        #     path = os.path.split(file)[0]
        #     shape = self.grids.shape
        #     to_remove_filename = self.grids.filename
        #     del(self.grids) 
        #     # os.remove(to_remove_filename)
        #     self.grids = np.memmap(
        #         os.path.join(path,s_config['filename']), 
        #         mode = self.config['mode'], 
        #         dtype = self.config['data_type'], 
        #         shape = shape
        #     )
        self._is_temp = False

    def configure_grid_name_map(self, config):
        """Configures the grid name map and sets in in config
        
        Paramaters
        ----------
        config:
            dict containing 'grid_names' a list 
        """
        grid_names = common.load_or_use_default(config, 'grid_names', [])
        if len(grid_names) > 0 and config['num_grids'] != len(grid_names):
            raise errors.GridNameMapConfigurationError(
                'Grid name list length does not equal N grids'
            )
        self.config['grid_name_map'] = {
            grid_names[i]: i for i in range(len(grid_names))
        }   

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

        if config['data_model'] == 'memmap':
            if filename is None:
                filename = os.path.join(mkdtemp(), 'temp.dat')
                self._is_temp = True
            # print('a', type(self), config['memory_shape'])
            grids = common.open_or_create_memmap_grid(
                filename, 
                config['mode'], 
                config['data_type'], 
                config['memory_shape']
            )
            # config['filename '] = filename

        else: # array
            grids = np.zeros(config['memory_shape'])
        return grids 
    
    def create_memory_shape (self, config):
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
        
    def create_real_shape (self, config):
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

        Raises
        ------
        MultigridFilterError:
            Raised if filter by `name` exists and force is false or if 
            shape of `data` is note equal to `grid_shape`
        """
        if name in self.filters and force == False:
            raise errors.MultigridFilterError("filters contains %s filter" % name)

        if data.shape != self.config['grid_shape']:
            raise errors.MultigridFilterError("filter shape does not match grid shape")
    
        self.filters[name] = data

    def activate_filter(self, name):
        """set a filter 

        Parameters
        ----------
        name: str or None
            if None: filters are unset
            else: filter is set based on name
        
        Raises
        ------
        MultigridFilterError
            When the filter name is invalid
        """
        if name in self.filters or name is None:
            self.current_filter = name
        else:
            raise errors.MultigridFilterError("invalid filter: %s" % name)
    
    def deactivate_filter(self):
        """set cuttent_filter to None
        """
        self.current_filter = None
        
    def lookup_grid_number(self, grid_id):
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
        return grid_id if type(grid_id) is int \
                       else self.config['grid_name_map'][grid_id]
    
    def lookup_grid_slice(self, start = None, end = None, step = None):
        """Get grid id slice from multigrid keys

        parameters
        ----------
        start: int or str
            start key
        end: int or str
            end key
        step: int
            step for slice

        Returns
        -------
        slice
        """
        start = self.lookup_grid_number(start) if start else None
        end = self.lookup_grid_number(end) if end else None
        step = step if step else 1

        return slice(start, end, step)

    def lookup_grid_range(self, start = None, end = None, step = 1):
        """Get grid id range from multigrid keys

        parameters
        ----------
        start: int or str
            start key
        end: int or str
            end key
        step: int
            step for slice

        returns
        -------
        slice
        """
        start = self.lookup_grid_number(start) if start else 0
        end = self.lookup_grid_number(end) if end else len(self.grids)
        step = step if step else 1
  
        return range(start, end, step)      

    def lookup_grid_numbers(self, grid_ids):
        """Find the grid numbers for a list, range or slice of grid_ids

        Parameters
        ----------
        grid_ids: list, range, or slice
            grid ids to find numbers for

        returns
        -------
        list or range
            Grid numbers for ids
        """
        if not type(grid_ids) in [slice, range]:
            grid_ids = [self.lookup_grid_number(gid) for gid in grid_ids]
        else:
            if grid_ids.start is None and  grid_ids.stop is None:
                grid_ids = self.lookup_grid_slice(
                    grid_ids.start, grid_ids.stop, grid_ids.step
                )
            else:
                grid_ids = self.lookup_grid_range(
                    grid_ids.start, grid_ids.stop, grid_ids.step
                )  
        return grid_ids

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
            Filter is applied if `set_filter` has been called
        """
        grid_id = self.lookup_grid_number(grid_id)



        _filter = self.current_filter
        _filter = self.filters[_filter].flatten() if _filter else 1
        # if _filter != 1:
        # print(grid_id)
        grid = self.grids[grid_id] * _filter
        # else:
        #     grid = self.grids[grid_id]
        if flat:
            return grid
        return grid.reshape(self.config['grid_shape'])

    def get_subgrid(self, grid_id, index, flat = True):
        """Get a part of a grid 
        
        Paramaters
        ----------
        grid_id: str or key
            if an int, it should be the grid number.
            if a str, it should be a grid name.
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        flat: bool, Default True
            if True return flat array

        Paramaters
        ----------
        np.array like
            1d if flat, 2d otherwise.
            Filter is applied if `set_filter` has been called
        """
        subgrid = self.get_grid(grid_id, False)[index]
        if flat:
            return subgrid.flatten()
        return subgrid

    def get_grids(self, grid_ids, flat = True, write_protect = True):
        """Get a set of grids
        
        Parameters
        ----------
        grid_ids: list, or range
            Items should be int or str Multigird key values
        flat: bool, defaults true
            If true, each grid is flattend and the returned value
            is 2D, otherwise returned value is 3D
        write_protect: bool
            convert possible memmaps to arrays to ensure data my not 
            be accidentally overwritten

        Returns
        -------
        np.array
            2d if flat, 3d otherwise.
            Filter is applied if `set_filter` has been called
        """    
        grid_ids = self.lookup_grid_numbers(grid_ids)

        _filter = self.current_filter
        if _filter:
            _filter = self.filters[_filter].flatten() #if _filter else 1
            grids = self.grids[grid_ids] * _filter
        else:
            
            grids = self.grids[grid_ids]# * _filter
            if write_protect:
                grids = np.array(grids)


        
        if flat:
            return grids
        rows, cols = self.config['grid_shape']
        return grids.reshape([len(grids), rows, cols])

    def get_subgrids(self, grid_ids, index, flat=True):
        """Get parts of a set of grids
        
        Parameters
        ----------
        grid_ids: list, or range
            Items should be int or str Multigird key values
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        flat: bool, defaults true
            If true, each grid is flattend and the returned value
            is 2D, otherwise returned value is 3D

        Returns
        -------
        np.array
            2d if flat, 3d otherwise.
            Filter is applied if `set_filter` has been called
        """
        # print(index)
        grids = self.get_grids(grid_ids, False, False)

        if type(index) is tuple and len(index) == 2:
            index = slice(None,None), index[0], index[1]
        elif type(index) is tuple and len(index) == 1:
            index = slice(None,None), index[0]
            # print(index)
        else:
            index = slice(None,None), index
        
        subgrids = np.array(grids[index])
        shape = subgrids.shape
        if flat and len(shape) == 3:
            return subgrids.reshape(shape[0], shape[1] * shape[2])
        return subgrids
        
    def set_grid(self, grid_id, new_grid):
        """Set a grid
         Parameters
        ----------
        grid_id: int or str
            if an int, it should be the grid number.
            if a str, it should be a grid name.
        new_grid: np.array like, or number
            Grid to set. must be able to reshape to grid_shape.
        """
        grid_id = self.lookup_grid_number(grid_id)

        try:
            self.grids[grid_id] = new_grid.flatten()
        except AttributeError:
            self.grids[grid_id][:] = new_grid
    
    def set_subgrid(self, grid_id, index, new_grid):
        """sets the values of part of a given grid

        Parameters
        ----------
        grid_id: int or str
            if an int,  it should be the grid number.
            if a str, it should be a grid name.
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        new_gird: 
            values that can be broadcast in to shape of index
        """
        grid_id = self.lookup_grid_number(grid_id)

        self.grids[grid_id].reshape(self.config['grid_shape'])[index] = new_grid
        
    def set_grids(self, grid_ids, new_grids):
        """Set a set of grids
        
        Parameters
        ----------
        grid_ids: list, or range
            Items should be int or str Multigird key values 
        new_grids: np.array like
            if new_girds.shape can be broadcast in to `grid_shape` 
            then there is a single grid to set for all keys provided
            otherwise there is a new grid for each
        """
        grid_ids = self.lookup_grid_numbers(grid_ids)

        shape =  self.config['grid_shape']
    
        for idx, grid_id in enumerate(grid_ids):
            ## if new_girds shape can be broadcast in to `grid_shape` 
            ## then there is a single grid to set for all keys provided
            ## otherwise there is a new grid for each
            if new_grids.shape == shape or new_grids.shape == (np.prod(shape),):
                self.grids[grid_id] = new_grids.flatten()
            else:
                self.grids[grid_id] = new_grids[idx].flatten()

    def set_subgrids(self, grid_ids, index, new_grids):
        """Set a set of grids
        
        Parameters
        ----------
        grid_ids: list, or range
            Items should be int or str Multigird key values 
        index: int, slice, list, range, np.array, or tuple
                containing two of any of those
            index that is within grid_shape
        new_grids: np.array like
            if new_girds.shape can be broadcast in to `grid_shape` 
            then there is a single grid to set for all keys provided
            otherwise there is a new grid for each
        """
        grid_ids = self.lookup_grid_numbers(grid_ids)
        if type(grid_ids) is slice:
            grid_ids = self.lookup_grid_range(grid_ids.start, grid_ids.stop, grid_ids.step)

        shape = self.config['grid_shape']
        sub_shape = self.grids[0].reshape(shape)[index].shape

        for idx, grid_id in enumerate(grid_ids):
            if new_grids.shape == sub_shape or \
                        (len(new_grids.shape) == 1 and \
                        new_grids.shape[0] == np.prod(sub_shape)):
                self.grids[grid_id].reshape(shape)[index] = \
                    new_grids.reshape(sub_shape)
            else:
                self.grids[grid_id].reshape(shape)[index] = \
                    new_grids[idx].reshape(sub_shape)

    def save_figure(
            self, grid_id, filename, figure_func=figures.default, figure_args={}, data=None
        ):
        """
        """
        if data is None:
            data = self[grid_id].astype(float)
        
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

        TODO: fix datatype
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

    def as_ml_features(self, grid, mask = None, train_range=None ):
        """Get the data in a way that can be used in ML methods

        TODO:fix get_range
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

    def clip_grids_translate(self, extent, temp_dir='.clip_temp/'):
        """Clip grids using gdal.Translate

        Parameters
        ----------
        extent: tuple
            (minX, maxY, maxX, minY)
        temp_dir: path
            path to store temp data at
        
        Returns
        -------
        type(self)
            a grid object of the same type as the current grid
        """
        from .tools import load_and_create
        os.makedirs(temp_dir)

        gdal_type = raster.numpy_type_lookup(self.grids.dtype)
        
        name = 'clipped'
        
        self.save_all_as_geotiff(temp_dir, **{'base_filename':'full'})

        files = sorted(glob.glob(os.path.join(temp_dir, 'full*.tif')))

        for idx, in_file in enumerate(files):
            out_name = os.path.split(in_file)[1]
            out_name = out_name.replace('full','clipped')
            out_file = os.path.join(temp_dir, out_name)
            print(idx, in_file, out_file)
            raster.clip_raster(in_file, out_file, extent, datatpye=gdal_type)
            os.remove(in_file)

        files = sorted(glob.glob(os.path.join(temp_dir,'%s*.tif' % name)))
        d, md = raster.load_raster(files[0])
        
        lp = {
            "method": 'tiff',
            "directory": temp_dir, # have to supply a directory
            "file_name_structure": '%s_*.tif' % name,
            "sort_func": sorted, 
            "verbose": True}
        cp = {
            'name': self.config['dataset_name'] + '- sub area', 
            'grid_names': list(self.config['grid_name_map'].keys()), 
            'start_timestep': 
                self.config['start_timestep'] if 'start_timestep' in self.config else None, 
            'raster_metadata': md
        }

        rv = load_and_create(lp, cp)
        rv.config['description'] = 'extent of area: ' + str(extent)
        files = sorted(glob.glob(os.path.join(temp_dir,'*')))
        for file in files:
            os.remove(file)

        os.rmdir(temp_dir)
        return rv 

    def clip_grids(self, extent, location_format="ROWCOL", verbose=False):
        """Clip the desired extent from the multigrid. Returns a new 
        Multigrid with the smaller extent.  

        This function is less accurate than clip_grids_translate which
        which should be used instead in most cases

        Parameters
        ----------
        extent: tuple
            4 tuple containing top left and bottom right coordinates to clip
            data to


            (row_tr, col_tr, row_bl, col_bl) if location format == "ROWCOL"
            (east_tr, north_tr, east_bl, north_bl) if location format == "GEO"
            (Long_tr, Lat_tr, Long_bl, Lat_bl) if location == "WGS84"
        location_format: str, default "ROWCOL"
            "ROWCOl", "GEO", or "WGS84" to indcate if locations are in
            pixel, map, or WGS84 format
        verbose: bool, default False
    
        Returns
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
        
        view = raster.zoom_box(
            data, copy.deepcopy(top_l), copy.deepcopy(bottom_r)
        )
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
            view.config['description'] = 'Unknown clipped to' + str(extent)
        
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
            zoom = raster.zoom_box(
                grid, copy.deepcopy(top_l), copy.deepcopy(bottom_r)
            )
            view.grids[idx][:] = zoom.flatten()

        view.config['mask'] = raster.zoom_box(
            self.config['mask'], top_l, bottom_r
        )

        view.config['raster_metadata'] = copy.deepcopy(raster_meta)
        view.config['raster_metadata']['transform'] = view_transform

        for filter in self.config['filters']:
            filter_data = self.filters[filter]
            filter_data = raster.zoom_box(
                filter_data, 
                copy.deepcopy(top_l), copy.deepcopy(bottom_r)
            )
            view.add_filter(filter, filter_data )

        return view

    def clip_to_shape(
            self, shape, name='subarea', temp_dir='./temp', warp_options = {}
        ):
        """Clip the grid to a shape (from a vector file) using gdal warp

        Parameters
        ----------
        shape: path
            path to vector file with shape to clip to
        name: str
            Name for files, and sub area
        temp_dir: path
            path to store temp data at
        warp_options:
            Options to pass to gdal warp

        Returns
        -------
        multigrid.Multigrid
        """
        from .tools import load_and_create
        os.makedirs(temp_dir)
    
        self.save_all_as_geotiff(temp_dir, **{'base_filename':'full'})

        files = sorted(glob.glob(os.path.join(temp_dir, 'full*.tif')))
        # return 
        for idx, in_file in enumerate(files):
            out_file = os.path.join(temp_dir, '%s_%i.tif' % (name, idx) )
            
            raster.clip_polygon_raster(in_file, out_file, shape, **warp_options)

            os.remove(in_file)

        files = sorted(glob.glob(os.path.join(temp_dir,'%s*.tif' % name)))
        md = raster.load_raster(files[0])[1]
        
        lp = {
            "method": 'tiff',
            "directory": temp_dir, # have to supply a directory
            "file_name_structure": '%s_*.tif' % name,
            "sort_func": sorted, 
            "verbose": False}
    
        cp = {
            'name': self.config['dataset_name'] + '- sub area: ' + name, 
            # 'description': self.config['description'] + '- sub area: ' + name, 
            'grid_names': list(self.config['grid_name_map'].keys()), 
            'start_timestep': 
                self.config['start_timestep'] if 'start_timestep' in self.config else None, 
            'raster_metadata': md
        }

        rv = load_and_create(lp, cp)
        
        files = sorted(glob.glob(os.path.join(temp_dir,'*')))
        for file in files:
            os.remove(file)

        os.rmdir(temp_dir)
        return rv 
        
    def zoom_to(
            self, location, radius=50, location_format="ROWCOL", verbose=False
        ):
        """zoom in to center location

        Parameters
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
    
        Returns
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
        """This gets the max values 
        """

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

    def calc_statistics_for(
            self, keys, stat_func=np.mean, axis = 0, flat=False
        ):
        """Calculate the statstics for a given substet of grids

        Parameters
        ----------
        keys: list, or range
            Items should be int or str Multigird key values 
        stat_func: function, default np.mean
            a function at operates on numpy arrays
        axis: int, default 0
            axis along which to apply function
        flat: bool, default False
            if True results are a flattened array
        
        Returns
        -------
        np.array:
            value from statistic function
        """
        data = self.get_grids(keys, flat)
        return stat_func(data, axis)

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

        Parameters
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

