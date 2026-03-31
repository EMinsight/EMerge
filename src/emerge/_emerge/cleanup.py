from .simstate import _GLOBAL_SIMSTATES, _GEOMANAGER
import gc
from .selection import SELECTOR_OBJ, _CALC_INTERFACE
import gmsh
from .solver import DEFAULT_ROUTINE
from .logsettings import LOG_CONTROLLER, DEBUG_COLLECTOR, _LOG_BUFFER
from .geometry import _GEOMANAGER, _GENERATOR
from loguru import logger
from .geo.pcb import _NAME_MANAGER
def cleanup() -> None:
    """ Cleanup all global states used in EMerge."""
    logger.info('Cleaning up EMerge global states')
       
    
    _GEOMANAGER.clear()
    _NAME_MANAGER.clear()
    _CALC_INTERFACE.clear()
    SELECTOR_OBJ.clear()
    DEFAULT_ROUTINE.reset(hard=True)
    _GLOBAL_SIMSTATES.clear()
    _GENERATOR.reset()
    _LOG_BUFFER.clear()
    
    gc.collect()
    if gmsh.isInitialized():
        gmsh.clear()
        gmsh.finalize()
        
    logger.info('Cleanup complete!')
    
    LOG_CONTROLLER.clear()
    DEBUG_COLLECTOR.clear()