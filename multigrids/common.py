"""
Common
"""

import numpy as np
import os

load_or_use_default = lambda c, k, d: c[k] if k in c else d

def open_or_create_memmap_grid(filename, mode, dtype, shape, verbose=False):
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
        if verbose:
            print('Creating New:', filename, dtype, shape)
        grids = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)           
        del(grids)
    elif verbose:
        print('Using existing: %s. Is This what you intended?' % filename)
        
    return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

def format_subgrid_index(index):
    """
    """
    if type(index) is tuple and len(index) == 2: ## 2 slices
        index =  slice(None), index[0], index[1]
    elif type(index) is tuple and len(index) == 1: # 1 slices
        index = slice(None),index[0],slice(None)
    elif type(index) is slice:
        index = slice(None), index, slice(None)
    else: ## index is array 
        index = slice(None,None), index
    return index


