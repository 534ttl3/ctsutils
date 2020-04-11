import numpy as np
from matplotlib.transforms import Bbox
import matplotlib.lines as lines

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def draw_axes_bbox(fig, ax):
    """
    Sometimes you may want to draw an axes and draw it's bbox too.
    tags: bounding box
    """
    points = ax.bbox.get_points()/fig.get_dpi()
    # the bbox is a rectangle
    p1 = get_fractional_coords(fig, points[0][0], points[0][1])
    p2 = get_fractional_coords(fig, points[1][0], points[1][1])

    # draw the rectangle of the bounding box
    # draw lines into the figure, without any axes
    # see https://matplotlib.org/3.1.0/gallery/pyplots/fig_x.html
    l1 = lines.Line2D([p1[0], p1[0]], [p1[1], p2[1]],
                      transform=fig.transFigure, figure=fig, linestyle="--")
    l2 = lines.Line2D([p1[0], p2[0]], [p2[1], p2[1]],
                      transform=fig.transFigure, figure=fig, linestyle="--")
    l3 = lines.Line2D([p2[0], p2[0]], [p2[1], p1[1]],
                      transform=fig.transFigure, figure=fig, linestyle="--")
    l4 = lines.Line2D([p1[0], p2[0]], [p1[1], p1[1]],
                      transform=fig.transFigure, figure=fig, linestyle="--")

    fig.lines.extend([l1, l2, l3, l4])


def get_fractional_coords(fig, x_abs, y_abs):
    """ to e.g. set the size of a figure in inches or draw lines
    between points with coordinates given in inches inside a figure,
    functions are only available for fractional coordinates [0, 1] in x and y
    direction of a 2d figure. This function converts the absolute values
    you want to set into these fractional figure coordiantes.
    """
    figsize = fig.get_size_inches()
    fig_width = figsize[0]
    fig_height = figsize[1]

    x_fr = x_abs/fig_width
    y_fr = y_abs/fig_height
    return np.array([x_fr, y_fr])


def set_axes_position_and_dimensions_in_inches(fig, ax, x_abs, y_abs, w_abs, h_abs):
    """ Sometimes, one wants to set all coorindates and widths/heights of
    a subplot just as absolute values, without having matplotlib resize the individual
    automatically.
    Just pass a figure fig, an axes ax, and absolute values for x and y coordinates and
    for width and height of your subplot (or in general, your axes object).
    """
    # fractional coordinates \in [0, 1] correspond to bottom/top and left/right border
    # of the figure
    # for an axes object,
    # fractional coordiantes (and dimensions) within a figure
    # can be set by ax.set_position([l,b,w,h])

    figsize = fig.get_size_inches()
    fig_width = figsize[0]
    fig_height = figsize[1]

    x_fr = x_abs/fig_width
    y_fr = y_abs/fig_height

    w_fr = w_abs/fig_width
    h_fr = h_abs/fig_height

    ax.set_position([x_fr,   # lbwh
                     y_fr,
                     w_fr,
                     h_fr])
