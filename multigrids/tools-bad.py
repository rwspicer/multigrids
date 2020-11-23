"""
Create Multigrid
---------

create_multigrid.py

"""
from . import multigrid
from . import grid
from . import temporal_grid
from . import temporal


import numpy as np
import os
from tempfile import mkdtemp
import yaml
import copy
import sys
import matplotlib.pyplot as plt

from spicebox import raster

import glob

class MultigridCreationError (Exception):
    """Raised if multigrid creation fails"""

def create(data, 
        name="", description = "", mode='r+', 
        mask = None, grid_names = [], data_model = 'memmap',
        filename = None, start_timestep = None, raster_metadata=None
    ):
    """Creates a Grid, MultiGrid, TemporalGrid, or TemporalMultiGrid, based
    on shape of data, and charateristics of other parameters passed.

    Parameters
    ----------
    data: np.array
        shape is (rows,cols), (n_grids,rows,cols), (n_timesteps,rows,cols), or
        (n_timesteps,n_grids,rows,cols).
    name: str, defaults ""
        name of data set
    description: str, defaults ""
        description of dataset
    node: str, defaults 'r+'
        mode to open data in 'r+', 'r', 'w+', or 'c'.
    mask: np.array, defaults None
    grid_names: list
    data_model: str
        indcates how multigrid stores data "memmap" or "array"
    filename: path, Defaults None
        File used as memmap if multigrid is in memmap mode
        if None a temp file is created and used
    start_timestep: int, defaults None
        if an int is provided, and the data has shape (n_grids,rows,cols)
        data will be treated as a grided time series with shape 
        (n_timesteps,rows,cols), and a TemporalMultigrid is created.
        Otherwise, a MultiGrid is created. This argument is ignored for 
        other data shapes
    raster_metadata: dict, or None
        metadata about geotransform/projection for creating tiff raster files 
        from MultiGrid data. If a dict is passed it must contain  'projection', 
        and 'transform' keys with values being a str (OpenGIS Well Known 
        Text strings) and Tuple (x_origin,pixel_width,rotation_x,
        y_origin,rotation_y,pixel height)

    


    Returns
    -------
    Grid, MultiGrid, TemporalGrid, or TemporalMultiGrid
    """
    if start_timestep is None:
        temporal_data = False
    else:
        temporal_data = True

    dimensions = len(data.shape)
    if 2 == dimensions:
        GridClass = grid.Grid
        args = data.shape
    elif 3 == dimensions and temporal_data:
        GridClass = temporal_grid.TemporalGrid
        args = (data.shape[1], data.shape[2], data.shape[0])
    elif 3 == dimensions and not temporal_data:
        GridClass = multigrid.MultiGrid
        args = (data.shape[1], data.shape[2], data.shape[0])
    elif 4 == dimensions:
        GridClass = temporal.TemporalMultiGrid
        args = (data.shape[2], data.shape[3], data.shape[1], data.shape[0])
    else:
        raise MultigridCreationError(
            "Cannot determine Which MultiGrid type to use from data"
        )

    mg = GridClass(
        *args, 
        data_type = data.dtype, 
        mode=mode,
        dataset_name = name,
        description = description,
        mask = mask,
        grid_names = grid_names,
        data_model = data_model,
        filename = filename,
        initial_data = data,
        start_timestep = start_timestep,
        )

    if not raster_metadata is None:
        mg.config['raster_metadata'] = raster_metadata
        
    return mg




def from_yaml (yaml_file):
    """
    create multigrid from yaml description
    """
    pass


def get_raster_metadata(in_file):
    """Gets the projection and transform from a geotiff raster 

    Parameters
    ----------
    in_file: path
        path to geotiff raster file.

    Returns
    -------
    dict
        contains raster 'projection', and 'transform' as values
    """
    data, md = raster.load_raster(in_file)
    return md


def tiffs_to_array (
        directroy, file_name_structure='*.tif', sort_func = sorted, 
        verbose = False
        ): 
    """reads a series of sorted tif files and stores data in an 3D array

    Parameters
    ----------
    directory: path
        directory containing tif files
    file_name_structure: str, default '*.tif'
        matching pattern passed to glob function to read tif files
    sort_func: function, defaults sorted
        sorting function, must return a list
    verbose: bool, defaults False
        if True debug messages are printed to console

    Returns
    -------
    np.array or None
        An array containing the data from the files. The shape is 
    N_files, rows, cols, and the N_file dimension is sorted according to the 
    sort_func used. None is returned if no files match file_name_structure
    pattern
    """
    path = os.path.join(directroy, file_name_structure)
    files = glob.glob(path)
    files = sort_func(files)

    shape = None
    array = None
    for ix, fi in enumerate(files):
        if verbose:
            print('Reading file:', os.path.split(fi)[1])
        grid, md = raster.load_raster(fi)
        if shape is None:
            rows, cols = grid.shape
            shape = len(files), rows, cols
            if verbose:
                print('\tArray shape:', shape)
            array = np.zeros(shape) - np.nan  #initialize to np.nan
        
        array[ix][:] = grid

    return array 

