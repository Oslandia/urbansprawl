"""Geometry handling module
"""

import numpy as np
from shapely.geometry import Point

def expand_grid_point(point, factor, xoffset, yoffset):
    """Expand a given georeferenced point by an integer factor, knowing the
    size of horizontal and vertical area

    Parameters
    ----------
    point : shapely.geometry.Point
        Raw point that must be expanded
    factor : int
        Number of new points to generate on each row/column
    xoffset : float
        Area width
    yoffset : float
        Area height
    Returns
    -------
    list
        List of shapely.geometry.Point that models the expanded grid
    """
    x, y = point.x, point.y
    x_range = 0.5 * (1 - 1.0/factor) * xoffset
    y_range = 0.5 * (1 - 1.0/factor) * yoffset
    xs = np.arange(x - x_range, x + xoffset/2.0, xoffset/factor)
    ys = np.arange(y - y_range, y + yoffset/2.0, yoffset/factor)
    points = []
    for x_ in xs:
        for y_ in ys:
            points.append(Point(x_, y_))
    return points
