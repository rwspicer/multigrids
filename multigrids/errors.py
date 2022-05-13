"""
Errors
-------

exception classes
"""

class GridSizeMismatchError(Exception):
    """GridSizeMismatchError"""

class IncrementTimeStepError (Exception):
    """Raised if grid timestep not found"""
    
class InvalidGridIDError (Exception):
    """Raised if grid timestep not found"""

class InvalidTimeStepError (Exception):
    """Raised if timestep is out of range"""

class ClipError(Exception):
    """Raised for errors in clip generation"""

class MultigridCreationError (Exception):
    """Raised if multigrid creation fails"""

class LoadDataMethodError (Exception):
    """Raised when method is not passed to load_and_create"""

class MultigridConfigError (Exception):
    """Raised if a multigrid class is missing its configuration"""

class MultigridIOError (Exception):
    """Raised during multigrid IO"""

class MultigridFilterError (Exception):
    """Raised during multigrid Filter ops"""

class GridNameMapConfigurationError (Exception):
    """Raised if the grid name map configuration fails"""
