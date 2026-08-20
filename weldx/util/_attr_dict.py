"""Attribute dictionary helper."""

from collections.abc import Mapping


class AttrDict(dict):
    """Dictionary subclass that allows attribute access to its keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, Mapping):
                self.__dict__[key] = AttrDict(value)
            else:
                self.__dict__[key] = value
