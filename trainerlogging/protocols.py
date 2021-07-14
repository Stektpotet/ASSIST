from typing import Protocol, Dict, Any


class TrainerLogger(Protocol):

    def init(self, *args, **kwargs):
        """Initialize logger"""

    def log(self, data: Dict[str, Any], step: int = None, commit: bool = None, sync: bool = None) -> None:
        """Logging a data instance"""

    def close(self) -> None:
        """closing the logger"""
