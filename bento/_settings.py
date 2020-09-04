from typing import Any, Union
from typing import Tuple, List
from pandarallel import pandarallel

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

    def __init__(self, n_cores=1, progress_bar=False):
        self._n_cores = n_cores
        self._progress_bar = progress_bar
        pandarallel.initialize(nb_workers=self._n_cores, progress_bar=self._progress_bar)

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
        pandarallel.initialize(nb_workers=self._n_cores, progress_bar=self._progress_bar)

    @property
    def progress_bar(self) -> bool:
        """
        Return whether progress bars for parallel computing are shown.
        """
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, show: bool):
        """
        Set whether to display progress bars.
        """
        _type_check(show, "progress_bar", bool)
        self._progress_bar = show
        pandarallel.initialize(nb_workers=self._n_cores, progress_bar=self._progress_bar)

settings = BentoConfig()
