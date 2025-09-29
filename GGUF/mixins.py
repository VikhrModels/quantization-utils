import logging
import os

from environment import get_environment_info

from models import Quant


class LoggerMixin:
    """Mixin for adding logging capabilities to classes"""

    def __init__(self, logger_name: str = __name__, *args, **kwargs):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(logging.INFO)
        super().__init__(*args, **kwargs)

    @property
    def logger(self):
        return self.__logger

    def info(self, msg, *args, **kwargs):
        self.__logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.__logger.debug(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log error message and raise RuntimeError"""
        self.__logger.error(msg, *args, **kwargs)
        raise RuntimeError(msg)

    def warning(self, msg, *args, **kwargs):
        self.__logger.warning(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log critical message and raise RuntimeError"""
        self.__logger.critical(msg, *args, **kwargs)
        raise RuntimeError(msg)

    def exception(self, msg, stack_info=True, *args, **kwargs):
        """Log exception with stack trace and re-raise"""
        self.__logger.exception(msg, stack_info=stack_info, *args, **kwargs)
        raise

    def log(self, level, msg, *args, **kwargs):
        self.__logger.log(level, msg, *args, **kwargs)


class ModelMixin:
    """Mixin for model-related operations"""

    def __init__(self, model_id, cwd=None, *args, **kwargs):
        self._model_id = model_id
        self._cwd = cwd or os.getcwd()
        self._environment = get_environment_info()
        super().__init__(*args, **kwargs)

    @property
    def cwd(self):
        return self._cwd

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    def get_model_name(self):
        """Extract model name from model_id"""
        return self._model_id.split("/")[-1]

    def get_model_dir(self):
        """Get directory path for the model"""
        return os.path.join(self._cwd, "models", self._model_id)

    def get_quantized_filepath(self, quant: Quant):
        """Get file path for quantized model"""
        return os.path.join(
            self.get_model_dir() + "-GGUF",
            f"{self.get_model_name()}-{quant.value}.gguf",
        )

    @property
    def environment(self):
        return self._environment

    @property
    def acceleration(self):
        return self._environment.get("acceleration", "cpu")

    def is_cpu_environment(self) -> bool:
        """Check if running in CPU-only environment"""
        return self.acceleration == "cpu"
