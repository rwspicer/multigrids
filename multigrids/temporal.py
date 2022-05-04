"""
Temporal Multigrid
------------------

a class for multivariable grided temporal data
"""
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
                self.num_timesteps = yaml.load(f, Loader=yaml.Loader)['num_timesteps']  
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
        # time        

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
        # return {grid_names[i]: i for i in range(len(grid_names))}   

    def grid_id_list(self):
        id_list = set()
        for key in self.config['grid_name_map']:
            id_list.add(key[1])
        return sorted(id_list)


    def find_memory_shape (self,config):
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

    def find_real_shape (self, config):
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
        # print(key)
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
        # print(key)
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
                return self.set_grids(timesteps, value)
            elif timesteps == slice(None):
                return self.set_grids(grids, value)
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
                    return self.set_grid(grid_ids[0], value)
                return self.set_grids(grid_ids, value)


        else: 
            if grids is None or grids == slice(None):
                return self.set_subgrids(timesteps, value)
            elif timesteps == slice(None):
                return self.set_subgrids(grids, value)
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
                    return self.set_subgrid(grid_ids[0], value)
                return self.set_subgrids(grid_ids, value)

   

    

    def convert_grid_numbers_to_index(self, grid_numbers):

        grid_numbers = np.array(grid_numbers)  
        return grid_numbers[:,0], grid_numbers[:,1]

    def find_grid_number(self, grid_id):
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
        
        return super().find_grid_number(grid_id)

    def find_grid_slice(self, start = None, end = None, step = None):
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
        return self.find_grid_range(start, end, step)


    def find_grid_range(self, start = None, end = None, step = 1):
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
            return super().find_grid_range(start, end, step)

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
                    range_list.append(self.find_grid_number(key))
                counter += 1
                if counter == step:
                    counter = 0
        return range_list

    def find_grid_numbers(self, grid_ids):
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
                return self.find_grid_number(grid_ids)
            if all([gid in  self.grid_id_list() for gid in grid_ids]):
                temp = []
                for grid_id in grid_ids:
                    for key in self.config['grid_name_map']:
                        if key[1] == grid_id:
                            temp.append(key)
                grid_ids = sorted(temp)

            grid_ids = [self.find_grid_number(gid) for gid in grid_ids]
            # grid_ids = self.convert_grid_numbers_to_index(grid_ids)
        else:
            if type(grid_ids.start) is int and type(grid_ids.start) is int:
                step = grid_ids.step
            else:
                step = 1
            grid_ids = self.find_grid_range(
                grid_ids.start, grid_ids.stop, step
            )  
        return grid_ids
    
    # def find_grid_id_at_all_timesteps(self, grid_id):
    #     """
    #     """
    #     range_list = []
    #     for key in list(self.config['grid_name_map'].keys()):
    #         if key[1] == grid_id:
    #             range_list.append(key)
    #     return range_list

    # def find_grid_number_at_all_timesteps(self, grid_id):
    #     """
    #     """
    #     range_list = []
    #     for key in self.find_grid_id_at_all_timesteps(grid_id):
    #         range_list.append(self.find_grid_number(key))
    #     return range_list

   
    def get_grid(self, grid_id, flat=True):
        """Get a grid given grid id pair (timestep, grid_id)
        """
        ## filters carry over from get_grids_at_timestep
        g_num = self.find_grid_number(grid_id)[1]
        return self.get_grids(grid_id[0], flat)[g_num]

    def get_subgrid(self, grid_id, index,  flat = True):
        """Get a subgrid given grid id pair (timestep, grid_id), and index
        """
        ## filters carry over from get_grid
        subgrid = self.get_grid(grid_id, False)[index]

        if flat:
            return subgrid.flatten() 
        return subgrid

    def get_grids(self, grid_ids, flat=True):
        """
        """
        grid_ids = self.find_grid_numbers(grid_ids)
        # print(grid_ids)
        if type(grid_ids) is list and \
                not all([type(i) is int for i in grid_ids]):
            grid_ids = self.convert_grid_numbers_to_index(grid_ids)

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
        """
        """
        if type(index) is tuple and len(index) == 2: ## 2 slices
            index = slice(None,None), index[0], index[1]
        elif type(index) is tuple and len(index) == 1: # 1 slices
            index = slice(None,None), index[0]
            # print(index)
        else: ## index is array 
            index = slice(None,None), index

        subgrids = self.get_grids(grid_id, False)[index]
        if flat:
            if len(subgrids.shape) == 2:
                return subgrids
            n_grids, rows, cols = subgrids.shape
            return subgrids.reshape([n_grids, rows * cols])
            
        return subgrids
 
    def get_grids_by_id(self, grid_id, flat=True):
        """
        """
        ## TODO add verification of id
        return self.get_grids(grid_id, flat)


    def get_subgrids_by_id(self, grid_id, index, flat=True):
        ## TODO add verification of id
        return self.get_grids(grid_id, index, flat)


    def get_grids_at_timesteps(self, timesteps, flat=True):
        """
        """
        ## TODO add verification of id
        return self.get_grids(timesteps, flat)

    def get_subgrids_at_timesteps(self, timesteps, index, flat=True):
        """
        """
        ## TODO add verification of id
        return self.get_subgrids(timesteps, index, flat)

    def get_grids_at_timestep(self, timestep, flat = True):
        """
        """
        ## TODO add verification of id
        return self.get_grids(timestep, flat)

    def get_subgrids_at_timestep(self, timestep, index,  flat = True):
        ## TODO add verification of id
        return self.get_subgrids(timestep, index, flat)

    def set_grid(self, grid_id, value):
        """Get a grid given grid id pair (timestep, grid_id)
        """
        
        g_num = self.find_grid_number(grid_id)
        if type(value) is np.ndarray:
            self.grids[g_num] = value.flatten()
        else:
            self.grids[g_num] = value

    def set_subgrid(self, grid_id, index,  value):
        """Get a subgrid given grid id pair (timestep, grid_id), and index
        """
        ## filters carry over from get_grid
        subgrid = self.set_grid(grid_id, False)[index]

        if flat:
            return subgrid.flatten() 
        return subgrid

    def set_grids(self, grid_ids, value):
        """
        """
        grid_ids = self.find_grid_numbers(grid_ids)
        # print(grid_ids)
        if type(grid_ids) is list and \
                not all([type(i) is int for i in grid_ids]):
            grid_ids = self.convert_grid_numbers_to_index(grid_ids)

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

    def set_subgrids(self, grid_id, index,  value):
        """
        """
        if type(index) is tuple and len(index) == 2: ## 2 slices
            index = slice(None,None), index[0], index[1]
        elif type(index) is tuple and len(index) == 1: # 1 slices
            index = slice(None,None), index[0]
            # print(index)
        else: ## index is array 
            index = slice(None,None), index

        subgrids = self.set_grids(grid_id, False)[index]
        if flat:
            if len(subgrids.shape) == 2:
                return subgrids
            n_grids, rows, cols = subgrids.shape
            subgrids.reshape([n_grids, rows * cols])
            

 
    def set_grids_by_id(self, grid_id, value):
        """
        """
        ## TODO add verification of id
        self.set_grids(grid_id, value)


    def set_subgrids_by_id(self, grid_id, index, value):
        ## TODO add verification of id
        self.set_grids(grid_id, index, value)


    def set_grids_at_timesteps(self, timesteps, value):
        """
        """
        ## TODO add verification of id
        self.set_grids(timesteps, value)

    def set_subgrids_at_timesteps(self, timesteps, index, value):
        """
        """
        ## TODO add verification of id
        self.get_subgrids(timesteps, index, value)

    def set_grids_at_timestep(self, timestep, value):
        """
        """
        ## TODO add verification of id
        self.set_grids(timestep, value)

    def set_subgrids_at_timestep(self, timestep, index, value):
        ## TODO add verification of id
        self.set_subgrids(timestep, index, value)




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
