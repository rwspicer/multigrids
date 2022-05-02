

import numpy as np
import os

class GridSizeMismatchError(Exception):
    """GridSizeMismatchError"""
    pass

class IncrementTimeStepError (Exception):
    """Raised if grid timestep not found"""

load_or_use_default = lambda c, k, d: c[k] if k in c else d


# TODO: probably delete these
def is_grid(key):
    return type(key) is str

def is_grid_with_range(key):
    return type(key) is tuple and len(key) == 2 and \
        type(key[0]) is str and type(key[1]) is slice

def is_grid_with_index(key):
    return type(key) is tuple and len(key) == 2 and \
        type(key[0]) is str and type(key[1]) is int

def is_grid_list(key):
    if type(key) is tuple:
        return np.array([type(k) is str for k in key]).all()
    return False

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
        ## if file does not exist; initialize and delete
        print(filename, dtype, shape)
        grids = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)           
        del(grids)
    return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
