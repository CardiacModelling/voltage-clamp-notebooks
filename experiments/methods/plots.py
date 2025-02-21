#
# Shared plotting functions
#
import datetime

import matplotlib
import numpy as np


def protocol_bands(ax, protocol, color='#f0f0f0', zorder=0, s=False):
    """
    Adds background bands on a Matplotib axes ``ax``, based on the Myokit
    protocol ``protocol``.
    """
    f = 1e-3 if s else 1
    for e in list(protocol)[::2]:
        ax.axvspan(e.start() * f, (e.start() + e.duration()) * f, color=color,
                   zorder=zorder)


def text_panels(fig, panels, headers=None, x=0.875, y=0.975):
    """
    Draws one or more text panels on the given figure.

    The argument ``panels`` should be a list with one entry per panel. Panels
    will be stacked vertically (top to bottom), with a one-line space
    separating two panels. The list entry for each panel should be another list
    where every entry is a line of text.

    If given, the optional argument ``headers`` should be a list of the same
    length as ``panels``, in which each entry is a single line of text that
    will be shown in bold at the top of each panel.
    """
    if len(panels) < 1:
        return

    if headers is not None:
        if len(headers) != len(panels):
            raise ValueError(
                f'List of headers ({len(headers)}) should have same length as'
                f' list of panels ({len(panels)}).')
    else:
        headers = [None] * len(panels)

    # Get height of a representative text
    t = fig.text(x, y, 'Zy0pWg')
    h = t.get_window_extent(renderer=fig.canvas.get_renderer()
        ).transformed(fig.transFigure.inverted()).height * 1.3
    t.remove()

    # Convert to lines and styles
    for header, panel in zip(headers, panels):
        if header is not None:
            t = fig.text(x, y, header, weight='bold')
            y -= h
        for line in panel:
            t = fig.text(x, y, line)
            y -= h
        y -= h

