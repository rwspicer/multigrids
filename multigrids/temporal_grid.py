from tkinter import N
from .multigrid import MultiGrid
import numpy as np
import yaml
# import figures
import os

from . import common
from . import clip

class TemporalGrid (MultiGrid):
    """ A class to represent a grid over a fixed period of time,
    Implemented using numpy arrays.

    Parameters
    ----------
    *args: list
        List of required arguments, containing exactly 3 items: # rows, 
        # columns, # time steps.
        Example call: mg = MultiGrid(rows, cols, n_timesteps)
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
        'grid_name_map': map of grid years to their indices.
    grids: TemporalMultiGrid data, np.memmap or np.ndarray  
    current_grids: grids at the current timestep
    """
    
    def __init__ (self, *args, **kwargs):
        """ Class initializer """

        if type(args[0]) is str:
            with open(args[0], 'r') as f:
                self.num_timesteps = yaml.load(f, Loader=yaml.Loader)['num_timesteps']  
            super(TemporalGrid , self).__init__(*args, **kwargs)
        else:
            self.num_timesteps = args[2]
            super(TemporalGrid , self).__init__(*args, **kwargs)
        
        self.config['num_timesteps'] = self.num_timesteps
        self.config['timestep'] = 0
        self.grid = self.grids[0]
        self.config['delta_timestep'] = "unknown"
        # self.config['start_timestep'] = 0

    def reset_grid_name_map(self, delta_timestep):

        delta_timestep = delta_timestep.lower()
        if delta_timestep == "year":
            self.config['delta_timestep'] = "year"
            sy = self.config['start_timestep']
            ny = self.config['num_timesteps']
            self.config["grid_name_map"] = {"%s" % (sy + y): y for y in range(ny)}

    def new(self, *args, **kwargs):
        """Does setup for a new TemporalGrid object
        
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
        ib = common.load_or_use_default(kwargs, 'start_timestep', 0)
        config['start_timestep'] = ib
        kwargs['grid_names'] = [str(i) for i in range(ib, ib + args[2])]
        mg_config, grids = super(TemporalGrid, self).new(*args, **kwargs)
        mg_config.update(config)
        return mg_config, grids

    def __getitem__(self, key): 
        """ Get item function
        
        Parameters
        ----------
        key: str, int, or tuple

        Returns
        -------
        np.array like
        """
        if type(key) in (str,):
            key = self.get_grid_number(key)
        else:
            # print (key)
            key -= self.config['start_timestep']
            
        return super().__getitem__(key)

    
    def set_grid(self, grid_id, new_grid):
        """Set a grid
         Parameters ffff
        ----------
        grid_id: int or str
            if an int, it should be the grid number.
            if a str, it should be a grid name.
        new_grid: np.array like, or any
            Grid to set. must be able to reshape to grid_shape.
        """

        if type(grid_id) is int:
            start = self.config['start_timestep']
            end =  start + self.config['num_timesteps']
            # print(grid_id) 
            if start <= grid_id < end:
                grid_id = grid_id - start
            else:
                raise IndexError('start_timestep <= timestep < end_timestep')
            # print(grid_id)
        super().set_grid(grid_id, new_grid)
        
    
    def set_sub_grid(self, grid_id, index, new_grid):
        """sets the values of part of a given grid

        Parameters
        ----------
        grid_id: int or str
            if an int, it should be a timestep
            if a str, it should be a grid name.
        index: slice of tuple of slices, or other index
            index into grid
        new_gird: 
            values that can be broadcast in to shape of index

        """
        if type(grid_id) is int:
            start = self.config['start_timestep']
            end =  start + self.config['num_timesteps']
            if start <= grid_id < end:
                grid_id = grid_id - start
            else:
                raise IndexError('start_timestep <= timestep < end_timestep')
            # print(grid_id)
        super().set_sub_grid(grid_id, index, new_grid)

  


    # def get_grids_at_keys(self,keys):
    #     """return the grids for the given keys

    #     Parameters
    #     ----------
    #     keys: list
    #         list of grids
        
    #     Returns
    #     -------
    #     np.array
    #     """
    #     select = np.zeros([len(keys), self.config['grid_shape'][0],self.config['grid_shape'][1] ] )
    #     c = 0
    #     for k in keys:
    #         select[c] = self[k]
    #         c += 1
    #     return select
        

    def get_memory_shape (self,config):
        """ Construct the shape needed for multigrid in memory from 
        configuration. 

        Parameters
        ----------
        config: dict
            Must have key 'grid_shape' a tuple of 2 ints

        Returns
        -------
        Tuple
            (num_timesteps, flattened shape of each grid ) 
        """ 
        return (
            self.num_timesteps, 
            config['grid_shape'][0] * config['grid_shape'][1]
        )

    def get_real_shape (self, config):
        """Construct the shape that represents the real shape of the 
        data for the MultiGird.

        Parameters
        ----------
        config: dict
            Must have key 'grid_shape' a tuple of 2 ints

        Returns
        -------
        Tuple
            (num_timesteps, 'rows', 'cols')
        """
        return (
            self.num_timesteps, 
            config['grid_shape'][0] , config['grid_shape'][1]
        )

    def increment_time_step (self):
        """increment time_step, 
        
        Returns 
        -------
        int 
            year for the new time step
        """
        # if archive_results:
        #     self.write_to_pickle(self.pickle_path)
        self.timestep += 1
        
        if self.timestep >= self.num_timesteps:
            self.timestep -= 1
            msg = 'The timestep could not be incremented, because the ' +\
                'end of the period has been reached.'
            raise common.IncrementTimeStepError(msg)
        self.grids[self.timestep][:] = self.grids[self.timestep-1][:] 
        self.grid = self.grids[self.timestep]
        
        return self.current_timestep()

    def save_clip(self, filename, clip_func=clip.default, clip_args={}):
        """
        """
    
        data = self.grids.reshape(self.config['real_shape'])
      
        try:
            clip_generated = clip_func(filename, data, clip_args)
        except clip.CilpError:
            return False
        return clip_generated
    
    def current_timestep (self):
        """gets current timestep adjused for start_timestep
        
        Returns
        -------
        int
            year of last time step in model
        """
        return self.config['start_timestep'] + self.config['timestep']

    def create_subset(self, subset_grids):
        """creates a multigrid containting only the subset_girds

        parameters
        ----------
        subset_grids: list
        """
        subset = super().create_subset(subset_grids)
        
        subset.config['start_timestep'] = subset_grids[0]
        subset.config['timestep'] = subset_grids[0]
        

        return subset

    def clip_grids(self, extent, location_format="ROWCOL", verbose=False):
        """
        Clip the desired extent from the multigrid. Returns a new 
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
        view = super().clip_grids(extent, location_format, verbose)
        view.config['start_timestep'] = self.config['start_timestep']
        view.config['timestep'] = self.config['start_timestep']
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
        multigrid.TemporalGrid
        """
        view = super().zoom_to(location, radius, location_format, verbose)
        view.config['start_timestep'] = self.config['start_timestep']
        view.config['timestep'] = self.config['start_timestep'] 

        return view
        