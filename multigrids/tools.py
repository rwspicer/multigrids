"""
MultiGrid Tools
---------------

Tools for creation/editing of multigrid classes

"""
import os
import shutil
from tempfile import mkdtemp
import glob

from multiprocessing import Process, active_children, cpu_count, Lock
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from spicebox import raster

from . import multigrid
from . import grid
from . import temporal_grid
from . import temporal

from . import errors

def create(data, 
        name="", description = "", mode='r+', 
        mask = None, grid_names = [], data_model = 'memmap',
        filename = None, start_timestep = None, raster_metadata=None,
        delta_timestep = None, shape=None, dtype=None, save_to = None,
        force_temporal = False
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
    if start_timestep is None and not force_temporal:
        temporal_data = False
    else:
        temporal_data = True

    no_data = data is None
    if no_data:
        dimensions = len(shape)
    else:  
        dimensions = len(data.shape)
        shape = data.shape

    if 2 == dimensions:
        GridClass = grid.Grid
        args = shape 
    elif 3 == dimensions and temporal_data:
        GridClass = temporal_grid.TemporalGrid
        args = (shape[1], shape[2], shape[0])
    elif 3 == dimensions and not temporal_data:
        GridClass = multigrid.MultiGrid
        args = (shape[1], shape[2], shape[0])
    elif 4 == dimensions:
        GridClass = temporal.TemporalMultiGrid
        args = (shape[2], shape[3], shape[1], shape[0])
    else:
        raise errors.MultigridCreationError(
            "Cannot determine Which MultiGrid type to use from data"
        )
    # print('xx', data.filename)
    # plt.imshow(data[0])
    # plt.show()
    mg = GridClass(
        *args, 
        data_type = data.dtype if (not data is None) else dtype, 
        mode=mode,
        dataset_name = name,
        description = description,
        mask = mask,
        grid_names = grid_names,
        data_model = data_model,
        filename = filename,
        initial_data = data,
        start_timestep = start_timestep,
        delta_timestep=delta_timestep,
        save_to = save_to
    )

    
    if not raster_metadata is None:
        mg.config['raster_metadata'] = raster_metadata

    mg._is_temp = True
    if not save_to is None:
        mg.save(save_to)
     
    return mg


def from_yaml (yaml_file):
    """
    create multigrid from yaml description
    """
    pass


def get_raster_metadata(raster_file):
    """Gets the projection and transform from a geotiff raster 

    Parameters
    ----------
    raster_file: path
        path to geotiff raster file.

    Returns
    -------
    dict
        contains raster 'projection', and 'transform' as values
    """
    md = raster.load_raster(raster_file)[1]
    return md


def tiffs_to_array (
        directory = None, file_name_structure='*.tif', sort_func = sorted, 
        verbose = False, precreated_grid = None, **kwargs
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
    **kwargs: 
        this does nothing, but allows load and create to 
        select from differently structure functions for loading data

    Returns
    -------
    np.array or None
        An array containing the data from the files. The shape is 
    N_files, rows, cols, and the N_file dimension is sorted according to the 
    sort_func used. None is returned if no files match file_name_structure
    pattern
    """
    if directory is None:
        raise IOError("directory dose not exist")
    if verbose:
        print(
            "Directory and wild card being used:", 
            directory, file_name_structure
        )

    if not precreated_grid:
        temp_file = os.path.join(directory, 'temp.mgdata')
        if 'filename' in kwargs:
            temp_file = kwargs['filename']
        shape = None
        array = None
    else:
        shape = precreated_grid.config['real_shape']
        array = precreated_grid.grids


    path = os.path.join(directory, file_name_structure)
    files = glob.glob(path)
    files = sort_func(files)
    if verbose:
        print ("Displaying First  15 sorted files:")
        for f in files[:15]:
            print('\t', f)
        print ("Displaying last  15 sorted files:")
        for f in files[-15:]:
            print('\t', f)

    
    for ix, fi in enumerate(files):
        if verbose:
            print('Reading file:', os.path.split(fi)[1])
        grid, md = raster.load_raster(fi)
        dtype = grid.dtype
        if shape is None:
            rows, cols = grid.shape
            shape = len(files), rows, cols
            if verbose:
                print('\tArray shape:', shape)

            ## TODO: add init data? like 0s or something
            ## TODO: add option to create as an array
            array = np.memmap(temp_file,
                shape=shape, dtype=dtype, mode='w+') 
            print('---->',array.filename)
                
        # print((grid == 0).all())
        
        array[ix][:] = grid if precreated_grid is None else grid.flatten()
        # plt.imshow(array[ix])
        # plt.show()
        del (grid)
    
    # array.flush()
    return array 

def mp_tiffs_to_array (
        directory = None, file_name_structure='*.tif', sort_func = sorted, 
        verbose = False, precreated_grid = None, **kwargs
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
    **kwargs: 
        this does nothing, but allows load and create to 
        select from differently structure functions for loading data

    Returns
    -------
    np.array or None
        An array containing the data from the files. The shape is 
    N_files, rows, cols, and the N_file dimension is sorted according to the 
    sort_func used. None is returned if no files match file_name_structure
    pattern
    """
    if directory is None:
        raise IOError("directory dose not exist")
    if verbose:
        print(
            "Directory and wild card being used:", 
            directory, file_name_structure
        )

    if 'n_cores' in kwargs:
        n_cores = min(kwargs['n_cores'], cpu_count())
    else:
        n_cores = cpu_count()

    path = os.path.join(directory, file_name_structure)
    files = glob.glob(path)
    files = sort_func(files)

    if not precreated_grid:
        grid, md = raster.load_raster(files[0])
        dtype = grid.dtype
        temp_file = os.path.join(directory, 'temp.mgdata')
        if 'filename' in kwargs:
            temp_file = kwargs['filename']
        
        rows, cols = grid.shape
        shape = len(files), rows, cols
        array = np.memmap(temp_file, shape=shape, dtype=dtype, mode='w+') 
        
    else:
        shape = precreated_grid.config['real_shape']
        array = precreated_grid.grids

    # temp_file = os.path.join(directory, 'temp.mgdata')
    # if 'filename' in kwargs:
    #     temp_file = kwargs['filename']

    

    if verbose:
        print ("Displaying First  15 sorted files:")
        for f in files[:15]:
            print('\t', f)
        print ("Displaying last  15 sorted files:")
        for f in files[-15:]:
            print('\t', f)

    # shape = None
    # grid, md = raster.load_raster(files[0])
    # dtype = grid.dtype

    # rows, cols = grid.shape
    # shape = len(files), rows, cols
    # if verbose:
    #     print('\tArray shape:', shape)
    # array = np.memmap(temp_file, shape=shape, dtype=dtype, mode='w+') 

    def __load_and_store(array, ix, fi, lock, precreated_grid):
        grid, md = raster.load_raster(fi)
        lock.acquire()
        array[ix][:] = grid if precreated_grid is None else grid.flatten()
        lock.release()
        del (grid)
    lock = Lock()
    for ix, fi in enumerate(files):
        if verbose:
            print('Reading file:', os.path.split(fi)[1])
        Process(target=__load_and_store, args=(
                array, ix, fi, lock, precreated_grid
            )
        ).start()
        while len(active_children()) > n_cores:
            continue
        
    return array 

def binary_to_array(
        directory = None, file_name_structure='*.bin', sort_func = sorted, 
        verbose = False, precreated_grid = None, **kwargs
    ):
    """
    """
    if directory is None:
        raise IOError("directory dose not exist")
    if verbose:
        print(
            "Directory and wild card being used:", 
            directory, file_name_structure
        )

    if not precreated_grid:
        temp_file = os.path.join(directory, 'temp.mgdata')
        if 'filename' in kwargs:
            temp_file = kwargs['filename']
        shape = (len(files), kwargs['rows'], kwargs['cols'])
        dtype = np.fromfile(files[0]).dtype 
        array = np.memmap(temp_file,
                shape=shape, dtype=dtype, mode='w+') 
    else:
        shape = precreated_grid.config['real_shape']
        array = precreated_grid.grids
    
    path = os.path.join(directory, file_name_structure)
    print('bta path',path)
    files = glob.glob(path)
    files = sort_func(files)


    if verbose:
        print ("Displaying First  15 sorted files:")
        for f in files[:15]:
            print('\t', f)
        print ("Displaying last  15 sorted files:")
        for f in files[-15:]:
            print('\t', f)

    

    for ix, fi in enumerate(files):
        grid =  np.fromfile(fi).reshape((kwargs['rows'], kwargs['cols']))
        array[ix][:] = grid
        del(grid)
    
    return array



def load_and_create( load_params = {}, create_params = {}):
    """loads data and creates a multigrid

    Parameters
    ----------
    load_params: dict
        parameters describing the data being loaded into a multigrid.
        
    create_params: dict 
        parameters describing the MultiGrid being created

    Returns
    -------
    Grid, MultiGrid, TemporalGrid, or TemporalMultiGrid
    """
    return create_and_load(load_params, create_params)


def create_and_load( load_params = {}, create_params = {}):
    """loads data and creates a multigrid

    Parameters
    ----------
    load_params: dict
        parameters describing the data being loaded into a multigrid.
        
    create_params: dict 
        parameters describing the MultiGrid being created

    Returns
    -------
    Grid, MultiGrid, TemporalGrid, or TemporalMultiGrid
    """
    if load_params['method'] == 'tiff':
        tiff_params = {
            "directory": None, # have to supply a directory
            "file_name_structure": '*.tif',
            "sort_func": sorted, 
            "verbose": False
        }
        tiff_params.update(load_params)
        load_params = tiff_params
        load_function = tiffs_to_array
        path = os.path.join(
            load_params['directory'], 
            load_params['file_name_structure']
        )
        files = glob.glob(path)
        grid, md = raster.load_raster(files[0])
        
        rows, cols = grid.shape
        create_params['shape'] = (
            len(files), rows, cols
        )
        create_params['dtype']  = grid.dtype
        create_params['raster_metadata'] = md

    elif load_params['method'] == 'mp_tiff':
        tiff_params = {
            "directory": None, # have to supply a directory
            "file_name_structure": '*.tif',
            "sort_func": sorted, 
            "verbose": False
        }
        tiff_params.update(load_params)
        load_params = tiff_params
        load_function = mp_tiffs_to_array

        path = os.path.join(
            load_params['directory'], 
            load_params['file_name_structure']
        )
        files = glob.glob(path)
        grid, md = raster.load_raster(files[0])
        
        rows, cols = grid.shape
        create_params['shape'] = (
            len(files), rows, cols
        )
        create_params['dtype']  = grid.dtype
        create_params['raster_metadata'] = md

    elif load_params['method'] == 'binary':
        ## need to have rows and cols load params
        bin_params = {
            "directory": None, # have to supply a directory
            "file_name_structure": '*.bin',
            "sort_func": sorted, 
            "verbose": False,
            
        }
        bin_params.update(load_params)
        load_params = bin_params
        load_function = binary_to_array
        path = os.path.join(
            load_params['directory'], 
            load_params['file_name_structure']
        )
        files = glob.glob(path)
        
        create_params['shape'] = (
            len(files), load_params['rows'], load_params['cols']
        )
        create_params['dtype']  = np.fromfile(files[0]).dtype 
    else:
        return False 
    
    # path = os.path.join(
    #     load_params['directory'], 
    #     load_load['file_name_structure']
    # )
    # files = glob.glob(path)
    
    # create_params['shape'] = (len(files), kwargs['rows'], kwargs['cols'])

    grid = create(None, **create_params)
    load_params['precreated_grid'] = grid
    load_function(**load_params)
    
    return grid
    
def combine(inputs, result_name, 
        final_extent = None, temp_dir = './TEMP-COMBINE', warp_options=[],
        datatype=raster.gdal.GDT_Float32):
    """combine geo-referenced multigrids into a single grid
    """
    os.makedirs(temp_dir)

    index = 0
    # temp_base_names = []
    # print(inputs.config)
    for grid in inputs:
        bn = 'mg-%i' % index
        # temp_base_names.append(bn)
        print(grid.config)
        grid.save_all_as_geotiff(temp_dir, **{'base_filename': bn})

    merged_path = os.path.join(temp_dir, 'merged')
    os.makedirs(merged_path)
    grid_names = sorted(list(inputs[0].config['grid_name_map'].keys()))
    # print(grid_names)
    
    for gn in grid_names:
        outfile = os.path.join(merged_path, 'merged-%s.tif', gn)
        matching_files = glob.glob(os.path.join(temp_dir, '*%s.tif' %  gn))
        raster.merge(matching_files, outfile, warp_options)
        if final_extent:
            raster.clip_raster(outfile, outfile, final_extent,  datatype)
    

    lp = {
        "method": 'tiff',
        "directory": merged_path,
        "file_name_structure": 'merged-*.tif',
        "sort_func": sorted, 
        "verbose": False
    }
    
    cp = {
        'name': result_name,  
        'grid_names': grid_names, 
        'start_timestep': 
            inputs[0].config['start_timestep'] if \
                'start_timestep' in inputs[0].config else None, 
        'raster_metadata': raster.load_raster(outfile)[1] # last outfile from previous loop should work
    }

    rv = load_and_create(lp, cp)
        
    shutil.rmtree(temp_dir)

    return rv 
        
