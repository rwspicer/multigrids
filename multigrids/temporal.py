"""
Temporal Multigrid
------------------

a class for multivariable grided temporal data
"""
from time import time
from .multigrid import MultiGrid
import numpy as np
import yaml
import os

from . import common, figures
import matplotlib.pyplot as plt

from . import clip

class TemporalMultiGrid (MultiGrid):
    """ A class to represent a set of multiple related grids of the same 
    dimensions, over a fixed period of time. Implemented using numpy arrays.

    Parameters
    ----------
    *args: list
        List of required arguments, containing exactly 3 items: # rows, 
        # columns, # grids, # time steps.
        Example call: mg = MultiGrid(rows, cols, n_grids, n_timesteps)
    **kwargs: dict
        Dictionary of key word arguments. Most of the valid arguments 
        are defined in the MultiGrid class, New and arguments with a different
        meaning are defined below:
    
    Attributes 
    ----------
    config: dict
         see MultiGrid attributes, and: 
        'grid_shape': 2 tuple, grid shape (rows, cols)
        'real_shape': 3 tuple, (num_grids, rows, cols)
        'memory_shape': 2 tuple, (num_grids, rows * cols)
        'num_timesteps': number of timesteps
        'timestep': the current timestep, for the grids in current_grids
        'start_timestep': the timestep to TemporalMultiGird start at. 
    grids: TemporalMultiGrid data, np.memmap or np.ndarray  
    current_grids: grids at the current timestep
    """
    
    def __init__ (self, *args, **kwargs):
        """ Class initializer """
        if type(args[0]) is str:
            with open(args[0], 'r') as f:
                self.num_timesteps = \
                    yaml.load(f, Loader=yaml.Loader)['num_timesteps']  
            super(TemporalMultiGrid , self).__init__(*args, **kwargs)
        else:
            self.num_timesteps = args[3]
            super(TemporalMultiGrid , self).__init__(*args, **kwargs)
            self.config['num_timesteps'] = self.num_timesteps
            # self.config['num_grids'] = self.num_timesteps
            self.config['timestep'] = 0
            self.config['start_timestep'] = \
                common.load_or_use_default(kwargs, 'start_timestep', 0)

        self.current_grids = self.grids[0]

    def new(self, *args, **kwargs):
        """Does setup for a new TemporalMultiGrid object
        
        Parameters
        ----------
        *args: list
            see MultiGird docs
        **kwargs: dict
            see MultiGird docs and:
                'start_timestep': int # to ues as the start timestep

        Returns
        -------
        Config: dict
            dict to be used as config attribute.
        Grids: np.array like
            array to be used as internal memory.
        """
        config = {}
        start = common.load_or_use_default(kwargs, 'start_timestep', 0)
        grid_names = common.load_or_use_default(
            kwargs, 'grid_names', list(range(self.num_timesteps))
        )
        config['start_timestep'] = start
        end = start + self.num_timesteps

        grid_names = [(ts, gn) for ts in range(start, end) for gn in grid_names]
        kwargs['grid_names'] = grid_names
        

        mg_config, grids = super().new(*args, **kwargs)
        mg_config.update(config)
        return mg_config, grids

    def create_grid_name_map(self, grid_names, config):
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
        gnm = {}
        ts = 0
        gn = 0
        for key in grid_names:
            gnm[key] = (ts, gn)
            gn += 1
            if gn >= config['num_grids']:
                gn = 0
                ts += 1

        required_size = config['num_grids'] * self.num_timesteps
        if len(gnm) > 0 and required_size != len(gnm):
            raise common.GridSizeMismatchError( 'grid name size mismatch' )
        
        return gnm

    def grid_id_list(self):
        """Lookup list of grids ids

        Returns
        -------
        list
            grid ids
        """
        id_list = set()
        for key in self.config['grid_name_map']:
            id_list.add(key[1])
        return sorted(id_list)

    def timestep_range(self):
        """get the range of time steps
        
        Returns
        -------
        range 
            range of timesteps
        """
        return range(
            self.config['start_timestep'], 
            self.config['start_timestep'] + self.config['num_timesteps']
        )

    def create_memory_shape (self,config):
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
        return (
            self.num_timesteps, config['num_grids'], 
            config['grid_shape'][0] * config['grid_shape'][1]
        )

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
        return (
            self.num_timesteps, config['num_grids'], 
            config['grid_shape'][0] , config['grid_shape'][1]
        )

    def  __getitem__(self, key): 
        """Get item function
        
        Parameters
        ----------
        key: int, tuple, list or range

        Returns
        -------
        np.array like
            Grid from the multigrid with the shape self.config['grid_shape']
        
        """
        index = None
        timesteps = None
        grids = None

        if type(key) in [range, slice, list, set, int, str]: ## slice, list, or single timestep
            key = [key]
        
        if len(key) > 4 and not all([type(i) in [int, str] for i in key]):
            
            raise IndexError ( 'Index has too many dimensions' )

        if 3 <= len(key) <=4:
            index = key[2:]
           

        if len(key) > 1:
            grids = key[1]
        timesteps = key[0]
        # print (timesteps, grids, index)


        if index is None: #whole grids

            if grids is None or grids == slice(None):
                return self.get_grids(timesteps, False)
            elif timesteps == slice(None):
                return self.get_grids(grids, False)
            else:
                if type(timesteps) in [str, int]:
                    timesteps = [timesteps]
                grid_ids = []
                if type(grids) in [str, int]:
                    grids = [grids]
                for ts in timesteps:
                    for gn in grids:
                        grid_ids.append((ts, gn))
                if len(grid_ids) == 1:
                    return self.get_grid(grid_ids[0], False)
                return self.get_grids(grid_ids, False)


        else: 
            if grids is None or grids == slice(None):
                return self.get_subgrids(timesteps, index, False)
            elif timesteps == slice(None):
                return self.get_subgrids(grids, index, False)
            else:
                if type(timesteps) in [str, int]:
                    timesteps = [timesteps]
                grid_ids = []
                if type(grids) in [str, int]:
                    grids = [grids]
                for ts in timesteps:
                    for gn in grids:
                        grid_ids.append((ts, gn))
                if len(grid_ids) == 1:
                    return self.get_subgrid(grid_ids[0], index, False)
                return self.get_subgrids(grid_ids, index, False)

    def  __setitem__(self, key, value): 
        """Set item function
        
        Parameters
        ----------
        key: int, tuple, list or range
        value:
            singel value or array that can be broadact into shape of key

        Returns
        -------
        np.array like
            Grid from the multigrid with the shape self.config['grid_shape']
        
        """
        index = None
        timesteps = None
        grids = None

        ## slice, list, or single timestep
        if type(key) in [range, slice, list, set, int, str]: 
            key = [key]
        
        if len(key) > 4 and not all([type(i) in [int, str] for i in key]):
            raise IndexError ( 'Index has too many dimensions' )

        if 3 <= len(key) <=4:
            index = key[2:]
           
        if len(key) > 1:
            grids = key[1]
        timesteps = key[0]

        if index is None: #whole grids
            if grids is None or grids == slice(None):
                self.set_grids(timesteps, value)
            elif timesteps == slice(None):
               self.set_grids(grids, value)
            else:
                if type(timesteps) in [str, int]:
                    timesteps = [timesteps]
                grid_ids = []
                if type(grids) in [str, int]:
                    grids = [grids]
                for ts in timesteps:
                    for gn in grids:
                        grid_ids.append((ts, gn))
                # print(grid_ids, len(grid_ids))
                if len(grid_ids) == 1:
                    self.set_grid(grid_ids[0], value)
                else:
                    self.set_grids(grid_ids, value)
        else: 
            if grids is None or grids == slice(None):
                self.set_subgrids(timesteps, index, value)
            elif timesteps == slice(None):
                self.set_subgrids(grids, index, value)
            else:
                if type(timesteps) in [str, int]:
                    timesteps = [timesteps]
                grid_ids = []
                if type(grids) in [str, int]:
                    grids = [grids]
                for ts in timesteps:
                    for gn in grids:
                        grid_ids.append((ts, gn))
                # print(grid_ids)
                if len(grid_ids) == 1:
                    self.set_subgrid(grid_ids[0], index, value)
                else:
                    self.set_subgrids(grid_ids, index, value)        

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
        if type(grid_id) is int:
            start = self.config['start_timestep']
            end =  start + self.config['num_timesteps']
            if start <= grid_id <= end:
                return grid_id - start
            else:
                raise IndexError('start_timestep <= timestep <= end_timestep')
        
        return super().lookup_grid_number(grid_id)

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
        step = step if step else 1
        return self.lookup_grid_range(start, end, step)

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
        if not type(start) is tuple:
            return super().lookup_grid_range(start, end, step)

        range_list = []
        in_range = False
        counter = 0
        for key in list(self.config['grid_name_map'].keys()):
            if key == start:
                in_range = True
            if key == end:
                in_range = False   
            if in_range:
                if counter == 0:
                    range_list.append(self.lookup_grid_number(key))
                counter += 1
                if counter == step:
                    counter = 0
        return range_list

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
        if grid_ids in self.grid_id_list():
            grid_ids = [grid_ids]
        
        if not type(grid_ids) in [slice, range]:

            ## This code may improve data access options
            # if type(grid_ids) in [list, tuple] and len(grid_ids) == 2 and \
            #         type(grid_ids[0]) in [slice, range, tuple, set, list] and \
            #             grid_ids[1] in self.grid_id_list() or \
            #             type(grid_ids[1]) in [tuple, set, list]:
            #     temp = []
            #     grid_names = grid_ids[1]
            #     if grid_names in self.grid_id_list():
            #         grid_names = [grid_names]
                
            #     for ts in grid_ids[0]:
            #         for name in grid_names:
            #             temp.append((ts, name))
            #     grid_ids = temp
            #     print(grid_ids)

            if type(grid_ids) is int:
                return self.lookup_grid_number(grid_ids)
            if all([gid in  self.grid_id_list() for gid in grid_ids]):
                temp = []
                for grid_id in grid_ids:
                    for key in self.config['grid_name_map']:
                        if key[1] == grid_id:
                            temp.append(key)
                grid_ids = sorted(temp)

            grid_ids = [self.lookup_grid_number(gid) for gid in grid_ids]
        else:
            # if type(grid_ids.start) is int and type(grid_ids.start) is int:
            step = grid_ids.step if  grid_ids.step  else 1
            # else:
            #     step = 
            grid_ids = self.lookup_grid_range(
                grid_ids.start, grid_ids.stop, step
            )  
        return grid_ids
    
    def get_grid(self, grid_id, flat=True):
        """Get a grid given grid id pair (timestep, grid_id)
        
        Parameters
        ----------
        grid_id: tuple
            tuple of a timestep and grid_id
        flat: bool, default True
            if flat data is returned as a flattened array

        Returns
        -------
        np.array 
            grid
        """
        ## filters carry over from get_grids_at_timestep
        g_num = self.lookup_grid_number(grid_id)[1]
        return self.get_grids(grid_id[0], flat)[g_num]

    def get_subgrid(self, grid_id, index,  flat = True):
        """Get a subgrid given grid id pair (timestep, grid_id), and index

        Parameters
        ----------
        grid_id: tuple
            tuple of a timestep and grid_id
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        flat: bool, default True
            if flat data is returned as a flattened array

        Returns
        -------
        np.array 
            part of a grid
        """
        ## filters carry over from get_grid
        subgrid = self.get_grid(grid_id, False)[index]

        if flat:
            return subgrid.flatten() 
        return subgrid
    
    def get_grids(self, grid_ids, flat=True):
        """Get a set of grids given grid_it range

        Parameters
        ----------
        grid_id: list, or range or, or int or str
            Keys for TemporalMultiGrid
        flat: bool, defaults true
            If true, each grid is flattend and the returned value
            is 2D, otherwise returned value is 3D

        Returns
        -------
        np.array
            2d if flat, 3d otherwise.
            Filter is applied if `set_filter` has been called
        """
        grid_ids = self.lookup_grid_numbers(grid_ids)
        # print(grid_ids)
        if type(grid_ids) is list and \
                not all([type(i) is int for i in grid_ids]):
            grid_ids = np.array(grid_ids)  
            grid_ids = grid_ids[:,0], grid_ids[:,1]
            
        _filter = self.current_filter
        _filter = self.filters[_filter].flatten() if _filter else 1
        # print(self.grids[grid_ids])
        grids = self.grids[grid_ids] * _filter

        if flat:
            if len(grids.shape) == 2:
                return grids
            else:
                return grids.reshape(
                    [grids.shape[0]* grids.shape[1], grids.shape[2]]
                )

        rows, cols = self.config['grid_shape']
       
        if len(grids.shape) == 2:
            return grids.reshape([grids.shape[0], rows, cols])
        else:
            return grids.reshape([grids.shape[0]*grids.shape[1], rows, cols])

    def get_subgrids(self, grid_id, index,  flat = True):
        """Get a set of subgrids given grid_it range

        Parameters
        ----------
        grid_id: list, or range or, or int or str
            Keys for TemporalMultiGrid
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
        index = common.format_subgrid_index(index)
        subgrids = self.get_grids(grid_id, False)[index]

        if flat:
            if len(subgrids.shape) <= 2:
                return subgrids

            n_grids, rows, cols = subgrids.shape
            return subgrids.reshape([n_grids, rows * cols]) 

        return subgrids
 
    def get_grids_by_id(self, grid_id, flat=True):
        """Gets all grids for a grid id

        Parameters
        ----------
        grid_id: int or str
            grid id in list returned by grid_id_list
        flat: bool
            if flat each grid is returned flattend otherwise grid shape is
            maintaine

        Returns
        -------
        np.array
            grids 
        """
        if grid_id in self.grid_id_list():
            return self.get_grids(grid_id, flat)

        raise common.InvalidGridIDError('Grid id not Valid:', grid_id)

    def get_subgrids_by_id(self, grid_id, index, flat=True):
        """Gets all subgrids for a grid id

        Parameters
        ----------
        grid_id: int or str
            grid id in list returned by grid_id_list
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        flat: bool
            if flat each grid is returned flattend otherwise index shape is
            maintained if possible

        Returns
        -------
        np.array
            grids 
        """
        if grid_id in self.grid_id_list():
            return self.get_grids(grid_id, index, flat)

        raise common.InvalidGridIDError('Grid id not Valid:', grid_id)

    def get_grids_at_timesteps(self, timesteps, flat=True):
        """Gets all grids for timesteps

        Parameters
        ----------
        grid_id: int or str
            grid id in list returned by grid_id_list
        flat: bool
            if flat each grid is returned flattend otherwise grid shape is
            maintaine

        Returns
        -------
        np.array
            grids 
        """
        for timestep in timesteps:
            if not timestep in self.timestep_range():
                raise common.TimestepsError('Timestep not Valid:', timesteps)

        return self.get_grids(timesteps, flat)

    def get_subgrids_at_timesteps(self, timesteps, index, flat=True):
        """Gets all subgrids for timesteps

        Parameters
        ----------
        grid_id: int or str
            grid id in list returned by grid_id_list
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        flat: bool
            if flat each grid is returned flattend otherwise index shape is
            maintained if possible

        Returns
        -------
        np.array
            grids 
        """
        for timestep in timesteps:
            if not timestep in self.timestep_range():
                raise common.TimestepsError('Timestep not Valid:', timesteps)
        return self.get_subgrids(timesteps, index, flat)

    def get_grids_at_timestep(self, timestep, flat = True):
        """Gets all grids for a timestep

        Parameters
        ----------
        grid_id: int or str
            grid id in list returned by grid_id_list
        flat: bool
            if flat each grid is returned flattend otherwise grid shape is
            maintained

        Returns
        -------
        np.array
            grids 
        """
        if not timestep in self.timestep_range():
            raise common.TimestepsError('Timestep not Valid:', timestep)
        return self.get_grids(timestep, flat)

    def get_subgrids_at_timestep(self, timestep, index,  flat = True):
        """Gets all subgrids for a timestep

        Parameters
        ----------
        grid_id: int or str
            grid id in list returned by grid_id_list
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        flat: bool
            if flat each grid is returned flattend otherwise index shape is
            maintained if possible

        Returns
        -------
        np.array
            grids 
        """
        if not timestep in self.timestep_range():
            raise common.TimestepsError('Timestep not Valid:', timestep)
        return self.get_subgrids(timestep, index, flat)

    def set_grid(self, grid_id, new_grid):
        """set a grid given grid id pair (timestep, grid_id)
        
        Parameters
        ----------
        grid_id: tuple
            tuple of a timestep and grid_id
        new_grid: np.array or value
            data to assign
        """
        g_num = self.lookup_grid_number(grid_id)
        if type(new_grid) is np.ndarray:
            self.grids[g_num] = new_grid.flatten()
        else:
            self.grids[g_num] = new_grid

    def set_subgrid(self, grid_id, index,  new_grid):
        """set a subgrid given grid id pair (timestep, grid_id), and index

        Parameters
        ----------
        grid_id: tuple
            tuple of a timestep and grid_id
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        new_grid: np.array or value
            data to assign
        """
        g_num = self.lookup_grid_number(grid_id)
        self.grids[g_num].reshape(self.config['grid_shape'])[index] = new_grid
       
    def set_grids(self, grid_ids, new_grids):
        """set a set of grids given grid_it range

        Parameters
        ----------
        grid_id: list, or range or, or int or str
            Keys for TemporalMultiGrid
        new_grids: np.array or value
            data to assign
        """
        grid_ids = self.lookup_grid_numbers(grid_ids)
       
        if type (grid_ids) is int:
            grid_ids = [grid_ids]

        shape =  self.config['grid_shape']

        for idx, grid_id in enumerate(grid_ids):
            # print (gid)
            if not hasattr(new_grids, 'shape'):
                self.grids[grid_id] = new_grids
            elif new_grids.shape == shape or \
                    new_grids.shape == (np.prod(shape),):
                self.grids[grid_id] = new_grids.flatten()
            else:
                # print(idx, grid_id)
                if type(grid_id) is int:
                    n_gids = len(self.grid_id_list())
                    for gix in range(n_gids):
                        data_id = idx*n_gids+gix
                        self.grids[grid_id][gix] = new_grids[data_id].flatten()
                else:
                    self.grids[grid_id] = new_grids[idx].flatten()

    def set_subgrids(self, grid_ids, index,  new_grids):
        """set a set of subgrids given grid_it range

        Parameters
        ----------
        grid_id: list, or range or, or int or str
            Keys for TemporalMultiGrid
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        new_grids: np.array or value
            data to assign
        """
        # print(index)
        # index = common.format_subgrid_index(index)
        # print(index)
        grid_ids = self.lookup_grid_numbers(grid_ids)
        if type (grid_ids) is int:
            grid_ids = [grid_ids]

        # shape = self.config['grid_shape']
        # idx_shape = self.get_subgrid(list(self.config['grid_name_map'].keys())[0], index, False).shape
        if all([type(gid) is int for gid in grid_ids]):
            grid_ids = [
                (gid,ix) for gid in grid_ids \
                         for ix in range(len(self.grid_id_list()))
            ]
        # print(grid_ids)
        # try:
        #     print('idx --->', new_grids.shape,idx_shape)
        # except:
        #     print(idx_shape)

        
       
        shape = self.config['grid_shape']
        sub_shape = self.grids[0,0].reshape(shape)[index].shape

        if not hasattr(new_grids, 'shape'):
            new_grids = np.zeros (sub_shape) + new_grids

        for idx, grid_id in enumerate(grid_ids):
            if new_grids.shape == sub_shape or \
                        (len(new_grids.shape) == 1 and \
                        new_grids.shape[0] == np.prod(sub_shape)):
                self.grids[grid_id].reshape(shape)[index] = \
                    new_grids.reshape(sub_shape)
            else:
                self.grids[grid_id].reshape(shape)[index] = \
                    new_grids[idx].reshape(sub_shape)

    def set_grids_by_id(self, grid_id, new_grid):
        """Sets a given grid at all time steps

        Parameters
        ----------
        grid_id: list, or range or, or int or str
            Keys for TemporalMultiGrid
        new_grid: np.array or value
            data to assign
        """
        if grid_id in self.grid_id_list():
            self.set_grids(grid_id, new_grid)
        raise common.InvalidGridIDError('Grid id not Valid:', grid_id)

    def set_subgrids_by_id(self, grid_id, index, new_grid):
        """Sets a given grid at all time steps

        Parameters
        ----------
        grid_id: list, or range or, or int or str
            Keys for TemporalMultiGrid
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        new_grid: np.array or value
            data to assign
        """
        if grid_id in self.grid_id_list():
            self.set_grids(grid_id, index, new_grid)
        raise common.InvalidGridIDError('Grid id not Valid:', grid_id)

    def set_grids_at_timesteps(self, timesteps, new_grid):
        """Sets all grids at given timesteps

        Parameters
        ----------
        timesteps: list, or range or, or int or str
            Keys for TemporalMultiGrid
        new_grid: np.array or value
            data to assign
        """
        for timestep in timesteps:
            if not timestep in self.timestep_range():
                raise common.TimestepsError('Timestep not Valid:', timesteps)
        self.set_grids(timesteps, new_grid)

    def set_subgrids_at_timesteps(self, timesteps, index, new_grid):
        """Sets all grids at given timesteps

        Parameters
        ----------
        timesteps: list, or range or, or int or str
            Keys for TemporalMultiGrid
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        new_grid: np.array or value
            data to assign
        """
        for timestep in timesteps:
            if not timestep in self.timestep_range():
                raise common.TimestepsError('Timestep not Valid:', timesteps)
        self.get_subgrids(timesteps, index, new_grid)

    def set_grids_at_timestep(self, timestep, new_grid):
        """Sets all grids at a given timestep

        Parameters
        ----------
        timesteps: list, or range or, or int or str
            Keys for TemporalMultiGrid
        new_grid: np.array or value
            data to assign
        """
        if not timestep in self.timestep_range():
            raise common.TimestepsError('Timestep not Valid:', timestep)
        self.set_grids(timestep, new_grid)

    def set_subgrids_at_timestep(self, timestep, index, new_grid):
        """Sets all grids at aa given timestep

        Parameters
        ----------
        timesteps: list, or range or, or int or str
            Keys for TemporalMultiGrid
        index: slice of tuple of slices, or other index
            index that is within grid_shape
        new_grid: np.array or value
            data to assign
        """
        if not timestep in self.timestep_range():
            raise common.TimestepsError('Timestep not Valid:', timestep)
        self.set_subgrids(timestep, index, new_grid)

    def increment_time_step (self, carry_data_forward = True):
        """Increment time_step, for current_girds.
        
        Parameters
        ----------
        carry_data_forward: Bool, default True
            If True data from previous timestep is set to data of current
            timestep when timestep is increpmented. This is the historic, and
            default behavior of the function.

        Returns 
        -------
        int 
            year for the new time step
        """
        self.config['timestep'] += 1
        
        if self.config['timestep'] >= self.num_timesteps:
            self.config['timestep'] -= 1
            msg = 'The timestep could not be incremented, because the ' +\
                'end of the period has been reached.'
            raise common.IncrementTimeStepError(msg)

        if carry_data_forward:
            self.grids[self.config['timestep']][:] = \
                self.grids[self.config['timestep']-1][:] 
        self.current_grids = self.grids[self.config['timestep']]
        
        return self.current_timestep()
    
    def current_timestep (self):
        """gets current timestep adjused for start_timestep
        
        Returns
        -------
        int
            year of last time step in model
        """
        return self.config['start_timestep'] + self.config['timestep']

    def save_figure(
            self, grid_id, ts, filename, figure_func=figures.default, figure_args={}, data=None
        ):
        """
        """
        if data is None:
            data = self[grid_id, ts].astype(float)
        # data[np.logical_not(self.AOI_mask)] = np.nan
        
        if not 'title' in figure_args:
            figure_args['title'] = self.dataset_name 
            if not grid_id is None:
                figure_args['title' ]+= ' ' + str( grid_id )
        fig = figure_func(data, figure_args)
        plt.savefig(filename)
        plt.close()

    def save_all_subgrid_figures(
            self, dirname, subgrid, figure_func=figures.default, figure_args={}, extension='.png'
        ):
        """
        """
        for ts in self.get_range():
            filename = os.path.join(
                dirname, 
                (self.config["dataset_name"]  + '_' + subgrid + '_' + str(ts) + extension).replace(' ','_')
            )
            figure_args['title'] = self.config["dataset_name"].replace('-', ' ')  + ' ' + str(ts)
            self.save_figure(subgrid, ts, filename, figure_func, figure_args)

    def show_figure(self, grid_id, ts, figure_func=figures.default, figure_args={}, data=None):
        """
        """
        if data is None:
            data = self[grid_id, ts].astype(float)
        # data[np.logical_not(self.AOI_mask)] = np.nan
        if not 'title' in figure_args:
            figure_args['title'] = self.config['dataset_name']
            if not grid_id is None:
                figure_args['title'] += ' ' + str( grid_id )
        fig = figure_func(data, figure_args)
        plt.show()
        plt.close()

    def save_clip(
            self, grid_id, filename, clip_func=clip.default, clip_args={}
        ):
        """
        """
        if not grid_id is None:
            data = self[grid_id]
            data = data.reshape(data.shape[0], data.shape[2], data.shape[3])
        else:
            data = None
        try:
            clip_generated = clip_func(filename, data, clip_args)
        except clip.CilpError:
            return False
        return clip_generated

    def get_as_ml_features(self, mask = None, train_range = None):
        """TemporalMultiGrid version of get_as_ml_features,

        Parameters
        ----------
        mask: np.array
            2d array of boolean values where true indicates data to get from
            the 2d array mask is applied to 
        train_range: list like, or None(default)
            [int, int], min an max years to get the features from the temporal 
            multigrid

        """
        features = [ [] for g in range(self.config['num_grids']) ]
        if mask is None:
            mask = self.config['mask']

        # print (mask)
        
        if train_range is None:
            train_range = self.get_range()
        # print (train_range)

        for ts in train_range:
            # print(ts)
            for grid, gnum in self.config['grid_name_map'].items():
                features[gnum] += list(self[grid, ts][mask])

            
        return np.array(features)

    def create_subset(self, subset_grids):
        """creates a multigrid containting only the subset_girds

        parameters
        ----------
        subset_grids: list
        """
        subset = super().create_subset(subset_grids)
        subset.config['start_timestep'] = self.config['start_timestep']
        subset.config['timestep'] = self.config['start_timestep']
        

        return subset
        

    def zoom_to(
            self, location, radius=50, location_format="ROWCOL", verbose=False
        ):
        """creates a multigrid containting only the subset_girds

        parameters
        ----------
        subset_grids: list
        """
        view = super().zoom_to(location, radius, location_format, verbose)
        view.config['start_timestep'] = self.config['start_timestep']
        view.config['timestep'] = self.config['start_timestep'] 

        return view

    def create_subset(self, subset_grids):
        """creates a multigrid containting only the subset_girds

        parameters
        ----------
        subset_grids: list
        """
        rows = self.config['grid_shape'][0]
        cols = self.config['grid_shape'][1]
        num_ts = self.config['num_timesteps']
        n_grids = len(subset_grids)
        subset = TemporalMultiGrid(rows, cols, n_grids, num_ts,
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

        subset.config['start_timestep'] = self.config['start_timestep']
        subset.config['timestep'] = self.config['timestep']
        
        for idx, grid in enumerate(subset_grids):
            subset[grid][:] = self[grid][:]

        return subset


def dumb_test():
    """Dumb unit tests, move to testing framework
    """
    g_names = ['first', 'second']

    init_data = np.ones([3,2,5,10])
    init_data[0,1] += 1
    init_data[1,0] += 2
    init_data[1,1] += 3
    init_data[2,0] += 4
    init_data[2,1] += 5

    t1 = TemporalMultiGrid(5, 10, 2, 3, grid_names = g_names, initial_data = init_data)
    t2 = TemporalMultiGrid(5, 10, 2, 3, grid_names = g_names)

    print( 't1 != t2:', (t1.grids != t2.grids).all() )
    print( 't1 == init_data:', (t1.grids == init_data.reshape(t1.memory_shape)).all() )
    print( 't1 == zeros:', (t2.grids == np.zeros([3,2,5*10])).all() )

    
    return t1
