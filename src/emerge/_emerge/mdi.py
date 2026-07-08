from __future__ import annotations
from typing import Any, Generator
from emsutil import Saveable
from copy import deepcopy

class MetaDataInstructions(Saveable):
    """An abstract class to communicate metadata between classes and objects."""
    
    def __init__(self):
        self.instructions: list[tuple[str, dict[str, Any]]] = []
        self.property: dict[str, Any] = dict()
        
    def store(self, name: str, value: str) -> None:
        """ Store a value under a name (dictionary like)"""
        self.property[name] = value
    
    def get(self, name) -> Any | None:
        """Get the value of a property or None if it doen't exist"""
        return self.property.get(name, None)
    
    def add(self, name: str, **kwargs) -> None:
        """Add a new metadata dictionary under a given name.
        

        Args:
            name (str): The name for the metadata caterogy.
        """
        self.instructions.append((name.strip().lower(), kwargs))
    
    def iter(self, name: str) -> Generator[dict[str,Any], None, None]:
        """Iterate through all metadatasets categorized under the same label name.

        Args:
            name (str): The name label of the metadata.

        Yields:
            Generator[dict[str,Any], None, None]: _description_
        """
        for iname, instr in self.instructions:
            if name.strip().lower() == iname:
                yield instr
    
    def copy(self) -> MetaDataInstructions:
        """Copy the metadata instruction set.

        Returns:
            MetaDataInstructions: _description_
        """
        mdi = MetaDataInstructions()
        mdi.instructions = deepcopy(self.instructions)
        mdi.property = self.property.copy()
        return mdi