import functools
import numpy as np
import os
import shutil
from datetime import datetime

DEBUG_MODE = 0


"""Cache Utils"""


def to_hashable(item, max_depth=5):
    """Converts a non hashable input into a hashable type for the purpose of using it as a part of the key in an
    instance's cache of calculated values."""
    if max_depth < 0:
        raise ValueError("Max recursion depth exceeded while trying to convert to hashable")

    if isinstance(item, dict):
        return tuple(sorted((k, to_hashable(v, max_depth - 1)) for k, v in item.items()))
    elif isinstance(item, (list, set)):
        return tuple(to_hashable(i, max_depth - 1) for i in item)
    elif isinstance(item, np.ndarray):
        return tuple(to_hashable(i, max_depth - 1) for i in item.tolist())
    elif hasattr(item, '__dict__'):  # Check if item is an object
        return id(item)  # return the memory address of the object
    else:
        return item


def cache_method(method):
    """
    Decorator that allows the results of a method's calculation to be stored in the instance cache.
    """

    if DEBUG_MODE == 2:
        return method

    cache_name = "_cache_" + method.__name__

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # If debug_mode is set to 2, the decorator will do nothing and just call the method.
        if self.data_opts.get('debug_mode') == 2:
            return method(self, *args, **kwargs)

        cache = getattr(self, cache_name, {})
        context_keys = (getattr(self, c).cache_id for c in ['neuron_type_context', 'data_type_context'] if
                        hasattr(self, c))
        cache_key = (id(self), context_keys, method.__name__, tuple(to_hashable(arg) for arg in args),
                     tuple(sorted(kwargs.items())))
        if cache_key not in cache:
            cache[cache_key] = method(self, *args, **kwargs)
            setattr(self, cache_name, cache)
        return cache[cache_key]

    return wrapper


def log_directory_contents(log_directory):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_subdirectory = os.path.join(log_directory, timestamp)
    os.makedirs(new_subdirectory, exist_ok=True)

    for item in os.listdir(current_directory):
        s = os.path.join(current_directory, item)
        d = os.path.join(new_subdirectory, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)



