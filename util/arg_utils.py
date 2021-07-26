from typing import Dict, Type, Union, Any, Callable, TypeVar, Tuple

__T = TypeVar("__T")


def filtered(config: Dict[str, Any], exclude: Tuple[str, ...]):
    filtered = config.copy()
    for k in exclude:
        if k in filtered:
            filtered.pop(k)
    return filtered


def configuration_dict(*configs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    merged = {}
    for cfg in (*configs, kwargs):
        for k, v in cfg.items():
            if isinstance(v, dict):
                merged[k] = configuration_dict(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged

def make_configuration(target: Union[Callable, Type], *args: Dict, **kwargs) -> Dict[str, Any]:
    """
    Make a configuration extracting default values from the calling type
    :param target:
    :param args:
    :param kwargs:
    :return:
    """
    is_class = isinstance(target, Type)
    fun = target.__init__ if is_class else target

    arg_count = fun.__code__.co_argcount - is_class  # don't count 'self'
    arg_names = fun.__code__.co_varnames[is_class:arg_count + is_class]

    full_config = configuration_dict(*args, kwargs)
    default_values = fun.__defaults__

    if default_values:
        full_config = configuration_dict({k: v for k, v in zip(arg_names[-len(default_values):], default_values)}, full_config)
    if len(full_config) < arg_count:
        raise UserWarning(f"Config does not supply enough parameters! Expected {arg_names}, got {full_config}")

    config = {k: v for k, v in full_config.items() if k in arg_names}
    if len(config) < arg_count:
        raise UserWarning(f"Config does not supply required parameters! Expected {arg_names}, got {full_config}")
    return config


def from_config(cls: Callable, *args: Dict, **kwargs) -> __T:
    accepted_args = make_configuration(cls, *args, **kwargs)
    try:
        return cls(**accepted_args)
    except TypeError as e:
        raise TypeError(e)

