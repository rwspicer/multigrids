from tkinter import N
from .multigrid import MultiGrid
import numpy as np
import yaml
# import figures
import os

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

from . import common, errors
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

        # print(self.config['delta_timestep'])
        self.configure_grid_name_map(kwargs)

        # self.config['delta_timestep'] = "unknown"
        # self.config['start_timestep'] = 0

    def convert_timesteps_to_julian_days(self):
        """Convert timesteps into number of days since start date

        Returns 
        -------
        list of ints
        """
        timesteps = list(self.config['grid_name_map'].keys())
        start = self.config['start_timestep']
        return [(ts - start).days for ts in timesteps]

    def configure_grid_name_map(self, config):
        """Configures the grid name map and sets in in config
        
        Parameters
        ----------
        config:
            dict containing delta_timestep which is a string, a relativedelta,
            or a dict with 'units' a string and 'delta' an int
        """
        try:
            delta_timestep = self.config['delta_timestep'] 
        except KeyError:
            #assume year based
            if 'grid_name_map' in self.config:
                key = list(self.config['grid_name_map'].keys())[0]
                if type(key) is str:
                    key = key.split('-')
                    year, month, day = 1066, 1, 1
                    if len(key) == 1:
                        delta_timestep = 'year'
                        year = int(key[0])
                    elif len(key) == 2:
                        delta_timestep = 'month'
                        year = int(key[0])
                        month = int(key[1])
                    elif len(key) == 3:
                        delta_timestep = 'day'
                        year = int(key[0])
                        month = int(key[1])
                        day = int(key[2])
                    else:
                        raise errors.GridNameMapConfigurationError(
                            'Delta Timestep could not be inferred'
                        )
                    try:
                        self.config['start_timestep'] = datetime(year,month,day)
                    except:
                        self.config['start_timestep'] = datetime(1066,month,day)
                else:
                    ## datetime
                    delta_timestep = list(self.config['grid_name_map'].keys())[1] - key
                    delta_timestep = relativedelta(delta_timestep)
            else:
                delta_timestep = 'year'

        if type(delta_timestep) is str:
            units = delta_timestep
            delta_timestep = 1
        elif type(delta_timestep) is dict:
            units = delta_timestep['units'].lower()
            delta_timestep = delta_timestep['delta']
        else:
            units = 'year' # defaults to year if ts is int

        if type(delta_timestep) is int:
            deltas = {
                "year":relativedelta(years=delta_timestep),
                "month":relativedelta(months=delta_timestep),
                "day":relativedelta(days=delta_timestep),
            }
            delta_timestep = deltas[units] if units else None
        
        if not type(delta_timestep) is relativedelta:
            raise errors.GridNameMapConfigurationError(
                'Delta Timestep could not be inferred'
            )

        
        sts = self.config['start_timestep']
        if type(sts) is int and units == 'year':
            delta_timestep = 1

        nts = self.num_timesteps

        self.config["grid_name_map"] = {
            sts + n * delta_timestep: n for n in range(nts)
        }
        # print(delta_timestep)
        self.config['delta_timestep'] = delta_timestep
        


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
        ib = common.load_or_use_default(kwargs, 'start_timestep', 1066)
        config['start_timestep'] = ib
        kwargs['grid_names'] = [str(i) for i in range(ib, ib + args[2])]
        mg_config, grids = super(TemporalGrid, self).new(*args, **kwargs)
        mg_config.update(config)
        return mg_config, grids

    def create_memory_shape (self,config):
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

    def create_real_shape (self, config):
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
        if type(grid_id) is str:
            # date, time grid_id.split('T') ## add times
            fields = grid_id.split('-')
            if  len(fields) > 1:
                year = int(fields[0])
                month = int(fields[1]) if len(fields) > 1 else 1
                day = int(fields[2]) if len(fields) > 2 else 1
            
                grid_id =  date(year, month, day)
            else:
                grid_id = int(grid_id)



        if type(grid_id) is int:
            start = self.config['start_timestep']
            if type(start) is int:
                end =  start + self.config['num_timesteps']
            else: #datetime
                end = start + self.config['num_timesteps'] * self.config['delta_timestep']
                grid_id = datetime(grid_id,1,1) ## only times this is an int if dts is a year
            if start <= grid_id <= end:
                if type(grid_id) is int:
                    return grid_id - start
                
            else:
                raise IndexError('start_timestep <= timestep <= end_timestep')               

        # print(grid_id)
        return super().lookup_grid_number(grid_id)
    
    def increment_time_step (self):
        """increment time_step, 
        
        Returns 
        -------
        int 
            year for the new time step
        """
        # if archive_results:
        #     self.write_to_pickle(self.pickle_path)
        self.config['timestep'] += 1
        
        if self.config['timestep'] >= self.num_timesteps:
            self.config['timestep'] -= 1
            msg = 'The timestep could not be incremented, because the ' +\
                'end of the period has been reached.'
            raise errors.IncrementTimeStepError(msg)
        self.grids[self.config['timestep']][:] = \
            self.grids[self.config['timestep']-1][:] 
        self.grid = self.grids[self.config['timestep']]
        
        return self.current_timestep()
    

    def timestep_range(self, step = 1):
        """get the range of time steps"""
        try:
            return range(
                self.config['start_timestep'], 
                self.config['start_timestep'] + self.config['num_timesteps'] *\
                    self.config['delta_timestep'],
                step
            )
        except:
            return sorted(list(self.config['grid_name_map'].keys()))
    
    def save_clip(self, filename, clip_func=clip.default, clip_args={}):
        """
        """
    
        data = self.grids.reshape(self.config['real_shape'])
      
        try:
            clip_generated = clip_func(filename, data, clip_args)
        except errors.ClipError:
            return False
        return clip_generated
    
    def current_timestep (self):
        """gets current timestep adjused for start_timestep
        
        Returns
        -------
        int
            year of last time step in model
        """
        return self.config['start_timestep'] + \
            self.config['timestep'] * self.config['delta_timestep']

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
        