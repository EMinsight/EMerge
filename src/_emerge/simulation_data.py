from __future__ import annotations
import numpy as np
from loguru import logger
from typing import TypeVar, Generic, Any, List, Union, Dict
from collections import defaultdict
from .solver import SolveReport

T = TypeVar("T")
M = TypeVar("M")

def assemble_nd_data(
    data_in: List[Union[float, np.ndarray]],
    vars_in: List[Dict[str, float]],
    axes: Dict[str, List[float]],
) -> np.ndarray:
    """
    Assemble a flat list of data entries into an N-dimensional array
    based on provided axis definitions.

    Parameters
    ----------
    data_in : List[float or np.ndarray]
        Flat list of data entries. Scalars are treated as shape ().
    vars_in : List[Dict[str, float]]
        List of dictionaries mapping variable names to their value for each data entry.
        Length must match data_in.
    axes : Dict[str, List[float]]
        Mapping from variable name to list of allowed values along that axis.
        If 'freq' is a key, its axis will be placed last.

    Returns
    -------
    np.ndarray
        An array of shape (*axis_lengths, *shp_out), where axis_lengths correspond
        to lengths of each axis in `axes` (with 'freq' last if present) and
        shp_out is the shape of the first ndarray in data_in or () if all scalars.
    """
    # Determine axis ordering
    axis_names = sorted(list(axes.keys()))
    if 'freq' in axis_names:
        axis_names.append(axis_names.pop(axis_names.index('freq')))
    axis_lengths = [len(axes[name]) for name in axis_names]

    # Determine output shape suffix
    # Assume all array entries share the same shape
    first = next((d for d in data_in if isinstance(d, np.ndarray)), None)
    shp_out = first.shape if isinstance(first, np.ndarray) else ()

    # Initialize output array
    out_shape = tuple(axis_lengths) + shp_out
    out = np.empty(out_shape, dtype=first.dtype if isinstance(first, np.ndarray) else float)

    # Fill in data
    for entry, var_map in zip(data_in, vars_in):
        # build index for each axis
        idx = tuple(
            axes[name].index(var_map[name])
            for name in axis_names
        )
        if isinstance(entry, np.ndarray):
            out[idx + tuple(slice(None) for _ in shp_out)] = entry
        else:
            out[idx] = entry

    return out

def generate_ndim(
    outer_data: dict[str, list[float]],
    inner_data: list[float],
    outer_labels: tuple[str, ...]
) -> np.ndarray:
    """
    Generates an N-dimensional grid of values from flattened data, and returns each axis array plus the grid.

    Parameters
    ----------
    outer_data : dict of {label: flat list of coordinates}
        Each key corresponds to one axis label, and the list contains coordinate values for each point.
    inner_data : list of float
        Flattened list of data values corresponding to each set of coordinates.
    outer_labels : tuple of str
        Order of axes (keys of outer_data) which defines the dimension order in the output array.

    Returns
    -------
    *axes : np.ndarray
        One 1D array for each axis, containing the sorted unique coordinates for that dimension, 
        in the order specified by outer_labels.
    grid : np.ndarray
        N-dimensional array of shape (n1, n2, ..., nN), where ni is the number of unique
        values along the i-th axis. Missing points are filled with np.nan.
    """
    # Convert inner data to numpy array
    values = np.asarray(inner_data)

    # Determine unique sorted coordinates for each axis
    axes = [np.unique(np.asarray(outer_data[label])) for label in outer_labels]
    grid_shape = tuple(axis.size for axis in axes)

    # Initialize grid with NaNs
    grid = np.full(grid_shape, np.nan, dtype=values.dtype)

    # Build coordinate arrays for each axis
    coords = [np.asarray(outer_data[label]) for label in outer_labels]

    # Map coordinates to indices in the grid for each axis
    idxs = [np.searchsorted(axes[i], coords[i]) for i in range(len(axes))]

    # Assign values into the grid
    grid[tuple(idxs)] = values

    # Return each axis array followed by the grid
    return (*axes, grid)

class PhysicsData:

    def __init__(self):
        self._variables: list[dict[str, float]] = []
        self._data: list[dict[str, Any]] = []
        self._reports: dict[tuple[tuple[str, float]], SolveReport] = dict()

    @property
    def last(self) -> dict:
        """The latest dataset entry that was added"""
        return self._data[-1]
    
    def setreport(self, reports: list[SolveReport], **variables) -> None:
        """Store a simulation report for a given simulation parameter setting

        Args:
            report (SolveReport): _description_
        """
        self._reports[tuple([(key, value) for key,value in variables.items()])] = reports

    def getreport(self, **variables) -> list[SolveReport] | None:
        """Return a simulation report for a given simulation setting

        Returns:
            SolveReport | None: _description_
        """
        self._reports.get(tuple([(key, value) for key,value in variables.items()]), None)

    def get(self, **variables: float) -> dict:
        """Return a dictionary entry for the given parameter settings to store information in

        Returns:
            dict: The data dictionary to be filled
        """
        dct = self.select(**variables)
        if dct is not None:
            return dct
        self._variables.append(variables)
        datadict = dict()
        self._data.append(datadict)
        return datadict
    
    def get_entry(self, index: int) -> dict:
        """Return the ith entry in the dataset. 

        Args:
            index (int): The data entry index

        Returns:
            dict: The data dictionary
        """
        return self._data[index]

    def select(self, **variables: float) -> dict | None:
        """Returns the first data dictionary that satisfies all variable specifications.

        Returns:
            dict: The data dictionary
        """
        for i, vrs in enumerate(self._variables):
            if all(vrs.get(k) == v for k, v in variables.items()):
                return self.get_entry(i)
        
        return None

class BaseDataset(Generic[T,M]):
    def __init__(self, datatype: T, matrixtype: M, scalar: bool):
        self._datatype: type[T] = datatype
        self._matrixtype: type[M] = matrixtype
        self._variables: list[dict[str, float]] = []
        self._data_entries: list[T] = []
        self._scalar: bool = scalar
        
        self._gritted: bool = None
        self._axes: dict[str, np.ndarray] = None
        self._ax_ids: dict[str, int] = None
        self._ids: np.ndarray = None
        self._gridobj: M = None

        self._data: dict[str, Any] = dict()
    
    def __getitem__(self, index: int) -> T:
        return self._data_entries[index]

    @property
    def _fields(self) -> list[str]:
        return self._datatype._fields
    
    @property
    def _copy(self) -> list[str]:
        return self._datatype._copy
    
    def store(self, key: str, value: Any) -> None:
        """Stores a variable with some value in the provided key. 
        Make sure that all values passed are picklable.

        Args:
            key (str): The name for the data entry
            value (Any): The value of the data entry
        """
        self._data[key] = value

    def load(self, key: str) -> Any | None:
        """Returns the data entry for a given key

        Args:
            key (str): The name of the data entry

        Returns:
            Any: The value of the data entry
        """
        return self._data.get(key, None)
    
    def get_entry(self, index: int) -> T:
        """Returns the physics dataset for  the given index

        Args:
            index (int): The index of the solution

        Returns:
            T: The Physics dataset
        """
        return self._data_entries[index]

    def select(self, **variables: float) -> T:
        """Returns the first physics dataset that satisfies the variable assignemnt.

        Returns:
            T: The physics dataset
        """
        index = next(
            i
            for i, var_map in enumerate(self._variables)
            if all(var_map.get(k) == v for k, v in variables.items())
        )
        return self.get_entry(index)
    
    def filter(self, **variables: float) -> list[T]:
        """Returns a list of all physics datasets that are valid for the given variable assignment

        Returns:
            list[T]: A list of all matching datasets
        """
        output = []
        for i, var_map in enumerate(self._variables):
            if all(var_map.get(k) == v for k, v in variables.items()):
                output.append(self.get_entry(i))
        return output
    
    def find(self, **variables: float) -> T:
        """Returns the physics dataset that is closest to the constraint given by the variables.

        Returns:
            T: The physics dataset.
        """
        output = []
        for i, var_map in enumerate(self._variables):
            error = sum([abs(var_map.get(k, 1e30) - v) for k, v in variables.items()])
            output.append((i,error))
        return self.get_entry(sorted(output, key=lambda x:x[1])[0][0])

    def axis(self, name: str) -> np.ndarray:
        """Returns a sorted list of all variables for the given name

        Args:
            name (str): The name of the variable axis

        Returns:
            np.ndarray: A sorted list of all unique values.
        """
        return np.sort(np.unique(np.array([var[name] for var in self._variables])))

    def new(self, **vars: float) -> T:
        """Creates a new dataset

        Returns:
            T: The physics dataset object
        """
        self._variables.append(vars)
        new_entry = self._datatype()
        self._data_entries.append(new_entry)
        return new_entry
    
    def _grid_axes(self) -> None:
        """This method attepmts to create a gritted version of the scalar dataset

        Returns:
            None
        """
        logger.debug('Attempting to grid simulation data')
        variables = defaultdict(set)
        for var in self._variables:
            for key, value in var.items():
                variables[key].add(value)
        N_entries = len(self._variables)
        N_prod = 1
        N_dim = len(variables)
        for key, val_list in variables.items():
            N_prod *= len(val_list)
        
        if N_entries == N_prod:
            logger.debug('Multi-dimensional grid found!')
            self._gritted = True
        else:
            logger.debug('Multi-dimensional grid not found')
            self._gritted = False
            return False
        
        self._axes = dict()
        self._ax_ids = dict()
        revax = dict()
        i = 0
        for key, val_set in variables.items():
            self._axes[key] = np.sort(np.array(list(val_set)))
            self._ax_ids[key] = i
            revax[i] = key
            i += 1
            
        axlist = []

        for idim in range(N_dim):
            axlist.append(self._axes[revax[idim]])
        
        indices = np.arange(N_entries)
        Ndimlist = [len(dim) for dim in axlist]
        self._ids = indices.reshape(Ndimlist)

        obj = self._matrixtype()

        axes_list = {key: list(self._axes[key]) for key in self._axes.keys()}
        for field in self._fields:
            data_field = [self._data_entries[i].__dict__[field] for i in range(N_entries)]
            # if isinstance(data[0], np.ndarray):
            #     shp = data[0].shape
            #     stacked = np.stack(data)   # shape (T, a, b)
            #     final_shape = tuple(Ndimlist) + shp
            #     result = stacked.reshape(*final_shape)
            #     obj.__setattr__(field, result)
            # else:
            #     matrix = np.array(data)[self._ids]
            #     obj.__setattr__(field, matrix)
            data_set = assemble_nd_data(data_field, self._variables, axes_list)
            obj.__setattr__(field, data_set)
        for copyfield in self._copy:
            
            obj.__setattr__(copyfield, self._data_entries[0].__dict__[copyfield])
        self._gridobj = obj

        return True
    
    @property
    def grid(self) -> M:
        """Returns the gridded version of the scalar dataset.

        Raises:
            ValueError: _description_

        Returns:
            M: The gritted physics dataset
        """
        if self._gritted is None:
            self._grid_axes()
        if self._gritted is False:
            logger.error('The dataset cannot be cast to a structured grid.')
            raise ValueError('Data not in regular grid')
        return self._gridobj