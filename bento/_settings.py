from typing import Any, Union
from typing import Tuple, List

def _type_check(var: Any, varname: str, types: Union[type, Tuple[type, ...]]):
    if isinstance(var, types):
        return
    if isinstance(types, type):
        possible_types_str = types.__name__
    else:
        type_names = [t.__name__ for t in types]
        possible_types_str = "{} or {}".format(
            ", ".join(type_names[:-1]), type_names[-1]
        )
    raise TypeError(f"{varname} must be of type {possible_types_str}")

class BentoConfig:
    """
    Config manager for bento.
    """
    
    def __init__(self, n_cores=2):
        self.n_cores = n_cores
        
    @property
    def n_cores(self) -> int:
        """
        Default number of cores to use for parallel computing.
        """
        return self._n_cores
    
    @n_cores.setter
    def n_cores(self, n_cores: int):
        _type_check(n_cores, "n_cores", int)
        self._n_cores = n_cores
        
settings = BentoConfig()