import logging
from rich.logging import RichHandler


class Settings:
    """
    Settings class for Bento.

    Parameters
    ----------
    verbosity : int
        Verbosity level for logging. Default is 0. See the logging module for more information (https://docs.python.org/3/howto/logging.html#logging-levels).
    log : logging.Logger
        Logger object for Bento.
    """

    def __init__(self, verbosity):
        self._verbosity = verbosity
        self._log = Logger(verbosity)

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._verbosity = value
        self._log.setLevel(value)

    @property
    def log(self):
        return self._log


class Logger:
    def __init__(self, verbosity):

        FORMAT = "%(message)s"
        logging.basicConfig(
            level=verbosity, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
        )
        self._logger = logging.getLogger("rich")

    def debug(self, text):
        self._logger.debug(text)

    def info(self, text):
        self._logger.info(text)

    def warn(self, text):
        self._logger.warning(text)


    def start(self, text):
        """
        Alias for self.info(). Start logging a method.

        Parameters
        ----------
        text : str
            Text to log.
        """
        self.info(f"[bold]{text}[/]")


    def step(self, text):
        """
        Alias for self.info(). Step logging.

        Parameters
        ----------
        text : str
            Text to log.
        """
        self._logger.info(text)


    def end(self, text):
        """
        End logging.

        Parameters
        ----------
        text : str
            Text to log.
        """
        self._logger.info(f"[bold]{text}[/]")

    def setLevel(self, value):
        """
        Set the verbosity level of the logger.

        Parameters
        ----------
        value : int
            Verbosity level for logging.
        """
        self._logger.setLevel(value)


settings = Settings(verbosity="WARNING")
