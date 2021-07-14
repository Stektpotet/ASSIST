import collections
from typing import Callable, TypeVar, Generic, Type

T0 = TypeVar('T0')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')

class EventV0:
    def __init__(self):
        self.__handlers = collections.OrderedDict()

    def __iadd__(self, handler: Callable):
        self.__handlers[handler] = None
        return self

    def __isub__(self, handler: Callable):
        self.__handlers.pop(handler)
        return self

    def __call__(self):
        for handler in self.__handlers:
            handler()

    def __len__(self):
        return len(self.__handlers)


class EventV1(Generic[T0]):
    def __init__(self):
        self.__handlers = collections.OrderedDict()

    def __iadd__(self, handler: Callable[[T0], None]):
        self.__handlers[handler] = None
        return self

    def __isub__(self, handler: Callable[[T0], None]):
        self.__handlers.pop(handler)
        return self

    def __call__(self, arg0: T0):
        for handler in self.__handlers:
            handler(arg0)

    def __len__(self):
        return len(self.__handlers)


class EventV2(Generic[T0, T1]):
    def __init__(self):
        self.__handlers = collections.OrderedDict()

    def __iadd__(self, handler: Callable[[T0, T1], None]):
        self.__handlers[handler] = None
        return self

    def __isub__(self, handler: Callable[[T0, T1], None]):
        self.__handlers.pop(handler)
        return self

    def __call__(self, arg0: T0, arg1: T1):
        for handler in self.__handlers:
            handler(arg0, arg1)

    def __len__(self):
        return len(self.__handlers)


class EventV3(Generic[T0, T1, T2]):
    def __init__(self):
        self.__handlers = collections.OrderedDict()

    def __iadd__(self, handler: Callable[[T0, T1, T2], None]):
        self.__handlers[handler] = None
        return self

    def __isub__(self, handler: Callable[[T0, T1, T2], None]):
        self.__handlers.pop(handler)
        return self

    def __call__(self, arg0: T0, arg1: T1, arg2: T2):
        for handler in self.__handlers:
            handler(arg0, arg1, arg2)

    def __len__(self):
        return len(self.__handlers)


class EventV4(Generic[T0, T1, T2, T3]):
    def __init__(self):
        self.__handlers = collections.OrderedDict()

    def __iadd__(self, handler: Callable[[T0, T1, T2, T3], None]):
        self.__handlers[handler] = None
        return self

    def __isub__(self, handler: Callable[[T0, T1, T2, T3], None]):
        self.__handlers.pop(handler)
        return self

    def __call__(self, arg0: T0, arg1: T1, arg2: T2, arg3: T3):
        for handler in self.__handlers:
            handler(arg0, arg1, arg2, arg3)

    def __len__(self):
        return len(self.__handlers)
