"""Color related tools."""

from typing import Dict, Generator, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

RGB_BLACK = 0x000000
RGB_BLUE = 0x0000FF
RGB_CYAN = 0x00FFFF
RGB_GREEN = 0x00AA00
RGB_MAGENTA = 0xFF00FF
RGB_RED = 0xFF0000
RGB_YELLOW = 0xAAAA00


def _color_rgb_to_int(rgb_color_tuple: Tuple[int, int, int]) -> int:
    """Convert an RGB color tuple to an 24 bit integer.

    Parameters
    ----------
    rgb_color_tuple : Tuple[int, int, int]
        The color as RGB tuple. Values must be in the range 0-255.

    Returns
    -------
    int :
        Color as 24 bit integer

    """
    return int("0x{:02x}{:02x}{:02x}".format(*rgb_color_tuple), 0)


def _color_int_to_rgb(integer: int) -> Tuple[int, int, int]:
    """Convert an 24 bit integer into a RGB color tuple with the value range (0-255).

    Parameters
    ----------
    integer : int
        The value that should be converted

    Returns
    -------
    Tuple[int, int, int]:
        The resulting RGB tuple.

    """
    return (integer >> 16) & 255, (integer >> 8) & 255, integer & 255


def _color_rgb_to_rgb_normalized(
    rgb: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    """Normalize an RGB color tuple with the range (0-255) to the range (0.0-1.0).

    Parameters
    ----------
    rgb : Tuple[int, int, int]
        Color tuple with values in the range (0-255)

    Returns
    -------
    Tuple[float, float, float] :
        Color tuple with values in the range (0.0-1.0)

    """
    return tuple([val / 255 for val in rgb])


def _color_rgb_normalized_to_rgb(
    rgb: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """Normalize an RGB color tuple with the range (0.0-1.0) to the range (0-255).

    Parameters
    ----------
    rgb : Tuple[float, float, float]
        Color tuple with values in the range (0.0-1.0)

    Returns
    -------
    Tuple[int, int, int] :
        Color tuple with values in the range (0-255)

    """
    return tuple([int(np.round(val * 255)) for val in rgb])


def color_int_to_rgb_normalized(integer):
    """Convert an 24 bit integer into a RGB color tuple with the value range (0.0-1.0).

    Parameters
    ----------
    integer : int
        The value that should be converted

    Returns
    -------
    Tuple[float, float, float]:
        The resulting RGB tuple.

    """
    rgb = _color_int_to_rgb(integer)
    return _color_rgb_to_rgb_normalized(rgb)


def _color_rgb_normalized_to_int(rgb: Tuple[float, float, float]) -> int:
    """Convert a normalized RGB color tuple to an 24 bit integer.

    Parameters
    ----------
    rgb : Tuple[float, float, float]
        The color as RGB tuple. Values must be in the range 0.0-1.0.

    Returns
    -------
    int :
        Color as 24 bit integer

    """
    return _color_rgb_to_int(_color_rgb_normalized_to_rgb(rgb))


def _shuffled_tab20_colors() -> List[int]:
    """Get a shuffled list of matplotlib 'tab20' colors.

    Returns
    -------
    List[int] :
        List of colors

    """
    num_colors = 20
    colormap = plt.cm.get_cmap("tab20", num_colors)
    colors = [colormap(i)[:3] for i in range(num_colors)]

    # randomize colors
    state = np.random.RandomState(42)
    state.shuffle(colors)

    return [_color_rgb_normalized_to_int(color) for color in colors]


def color_to_rgb_normalized(
    color: Union[int, Tuple[int, int, int], Tuple[float, float, float]]
) -> Tuple[float, float, float]:
    """Convert an arbitrary RGB color representation into a normalized RGB triplet.

    Parameters
    ----------
    color : Union[int, Tuple[int, int, int], Tuple[float, float, float]]
        A 24 bit integer, a triplet of integers with a value range of 0-255
        or a triplet of floats with a value range of 0.0-1.0
        that represent an RGB color.

    Returns
    -------
    Tuple[float, float, float] :
        RGB color triplet with the value range 0.0 to 1.0

    """
    if isinstance(color, Tuple) and len(color) == 3:
        if all(isinstance(number, int) for number in color):
            return _color_rgb_to_rgb_normalized(color)
        if all(isinstance(number, (int, float)) for number in color):
            return color
    if isinstance(color, int):
        return color_int_to_rgb_normalized(color)
    raise TypeError("Unsupported color format.")


_color_list = [
    RGB_RED,
    RGB_GREEN,
    RGB_BLUE,
    RGB_YELLOW,
    RGB_CYAN,
    RGB_MAGENTA,
    *_shuffled_tab20_colors(),
]


def color_generator_function() -> int:
    """Yield a 24 bit RGB color integer.

    The returned value is taken from a predefined list.

    Yields
    ------
    int:
        24 bit RGB color integer

    """
    while True:
        for color in _color_list:
            yield color


def get_color(key: str, color_dict: Dict[str, int], color_generator: Generator) -> int:
    """Get a 24 bit RGB color from a dictionary or generator function.

    If the provided key is found in the dictionary, the corresponding color is returned.
    Otherwise, the generator is used to provide a color.

    Parameters
    ----------
    key :
        The key that should be searched for in the dictionary
    color_dict :
        A dictionary containing name to color mappings
    color_generator :
        A generator that returns a color integer

    Returns
    -------
    int :
        RGB color as 24 bit integer

    """
    if color_dict is not None and key in color_dict:
        return _color_rgb_to_int(color_dict[key])
    try:
        return next(color_generator)
    except StopIteration:
        raise RuntimeError(f"given generator {color_generator} exhausted.")
