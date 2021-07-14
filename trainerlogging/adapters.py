from datetime import datetime
from typing import Dict, Any

import wandb
from .loggers import DictionaryLogger


class LoggerAdapter:
    def __init__(self, name: str = "Unnamed"):
        self.name = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{name}"

    def init(self, *args, **kwargs):
        pass

    def log(self,
            data: Dict[str, Any],
            step: int = None,
            commit: bool = None,
            sync: bool = None) -> None:
        pass

    def close(self):
        pass


class WeightsAndBiasesLoggerAdapter(LoggerAdapter):
    def __init__(self, name: str = "Unnamed", **kwargs):
        super().__init__(name)
        self.__kwargs = kwargs

    def init(self, *args, **kwargs):
        wandb.init(**self.__kwargs, name=self.name)
        wandb.watch(models=kwargs['model'], criterion=kwargs['loss_fn'])

    def log(self,
            data: Dict[str, Any],
            step: int = None,
            commit: bool = None,
            sync: bool = None) -> None:
        wandb.log(data, step, commit, sync)

    def close(self):
        wandb.finish()


class DictionaryLoggerAdapter(LoggerAdapter):
    def __init__(self, name: str = "Unnamed", logging_directory="."):
        super().__init__(name)
        self.__logger = DictionaryLogger(self.name)
        self.__logging_directory = logging_directory

    def log(self,
            data: Dict[str, Any],
            step: int = None,
            commit: bool = None,
            sync: bool = None) -> None:
        self.__logger.write(step=step, commit=commit, sync=sync, **data)

    def close(self):
        self.__logger.save(self.__logging_directory)


class DualLoggerAdapter(LoggerAdapter):
    def __init__(self, name: str = "Unnamed", logging_directory=".", **kwargs):
        super().__init__(name)
        self.__kwargs = kwargs
        self.__logger = DictionaryLogger(self.name)
        self.__logging_directory = logging_directory

    def init(self, *args, **kwargs):
        wandb.init(**self.__kwargs, name=self.name)
        wandb.watch(models=kwargs['model'], criterion=kwargs['loss_fn'])

    def log(self,
            data: Dict[str, Any],
            step: int = None,
            commit: bool = None,
            sync: bool = None) -> None:
        self.__logger.write(step=step, commit=commit, sync=sync, **data)
        wandb.log(data, step, commit, sync)

    def close(self):
        wandb.finish()
        self.__logger.save(self.__logging_directory)
