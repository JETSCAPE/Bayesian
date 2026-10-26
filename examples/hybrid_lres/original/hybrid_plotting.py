"""Shared publication styling. No inference, data loading, or statistical operations."""

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

DEFAULT_ELL = .2
DEFAULT_COLOR = '#007f86'
FIGURE_WIDTH = 7.15
LRES_LABEL = r'$\pi T L_{\mathrm{res}}$'
KAPPA_LABEL = r'$\kappa_{\mathrm{sc}}$'
ELL_LABEL = r'$\ell$'
RAA_LABEL = r'$R_{AA}$'

LENGTH_NORM = colors.LogNorm(.02, 1)
LENGTH_ANCHORS = ((.02, '#D58500'), (DEFAULT_ELL, DEFAULT_COLOR), (1., '#332288'))
LENGTH_CMAP = colors.LinearSegmentedColormap.from_list(
    'amber_teal_indigo', [(float(LENGTH_NORM(x)), c) for x, c in LENGTH_ANCHORS], N=4096)

_COMPARISONS = {
    'default': dict(color=DEFAULT_COLOR, linestyle='-', marker='D'),
    'blue': dict(color='#0072B2', linestyle=':', marker='o'),
    'vermillion': dict(color='#D55E00', linestyle='--', marker='s'),
    'purple': dict(color='#CC79A7', linestyle='-.', marker='^'),
}
_SELECTION_ROLES = {'raa_hpt30_full': 'blue',
                    'inclusive_plus_only_alice_tg': 'vermillion',
                    'raa_dr12': 'purple', 'raa_theta_dr12': 'default'}
_FONT_SIZES = {'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 12,
               'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12}


def comparison_style(role='default'):
    """Return a fresh categorical style, independent of series order."""
    role = _SELECTION_ROLES.get(role, role)
    return dict(_COMPARISONS[role])


def length_style(ell):
    """Finite lengths share one fixed log color scale; limits use neutral dashes."""
    ell = float(ell)
    if np.isnan(ell) or ell < 0:
        raise ValueError('ell must be nonnegative and not NaN')
    if ell == 0:
        return dict(color='#989898', linestyle=':')
    if np.isinf(ell):
        return dict(color='#252525', linestyle='--')
    if not LENGTH_NORM.vmin <= ell <= LENGTH_NORM.vmax:
        raise ValueError('Finite ell must lie in [0.02, 1]; do not silently clip colors')
    return dict(color=DEFAULT_COLOR if ell == DEFAULT_ELL else colors.to_hex(LENGTH_CMAP(LENGTH_NORM(ell))),
                linestyle='-')


def _rc(font_scale):
    if not np.isfinite(font_scale) or font_scale <= 0:
        raise ValueError('font_scale must be positive and finite')
    # Matplotlib ships genuine Computer Modern fonts; no local TeX install needed.
    font = Path(mpl.get_data_path()) / 'fonts/ttf/cmr10.ttf'
    if not any(f.name == 'cmr10' for f in font_manager.fontManager.ttflist):
        font_manager.fontManager.addfont(font)
    rc = {
        'font.family': 'cmr10', 'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.edgecolor': '#202020', 'axes.linewidth': 1.5, 'axes.grid': False,
        'text.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'auto',
        'xtick.color': '#303030', 'ytick.color': '#303030',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.major.width': 1.15, 'ytick.major.width': 1.15,
        'xtick.minor.size': 2.8, 'ytick.minor.size': 2.8,
        'xtick.minor.width': .85, 'ytick.minor.width': .85,
        'lines.linewidth': 2.6, 'legend.frameon': False, 'legend.handlelength': 2.3,
        'pdf.fonttype': 42, 'svg.fonttype': 'path', 'savefig.dpi': 300,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'figure.figsize': (FIGURE_WIDTH, 4.4),
        'text.usetex': False,
    }
    rc.update({key: value * font_scale for key, value in _FONT_SIZES.items()})
    return rc


def apply_style(*, font_scale=1.30):
    """Set approved defaults for figures created subsequently."""
    mpl.rcParams.update(_rc(font_scale))


@contextmanager
def paper_style(*, font_scale=1.30):
    """Apply approved defaults temporarily; restore the caller's rcParams."""
    with mpl.rc_context(_rc(font_scale)):
        yield


def frame_axes(ax, minor=True):
    """Full frames; optionally install midpoint minor ticks on linear axes only."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_position(('outward', 0))
        spine.set_linewidth(1.5)
    if minor:
        if ax.get_xscale() == 'linear':
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        if ax.get_yscale() == 'linear':
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='minor', length=2.8, width=.85)


def style_figure(fig, *, legacy_text_scale=1., legacy_line_scale=1.):
    """Frame an existing figure; optional one-time scaling for legacy plotters.

    New plots should use paper_style instead. Reapplying this with the same
    scales is a no-op, so repeated exports cannot progressively enlarge a plot.
    """
    scales = (legacy_text_scale, legacy_line_scale)
    if any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Scaling factors must be positive and finite')
    previous = getattr(fig, '_hybrid_plotting_scales', None)
    if previous is not None:
        if previous != scales:
            raise ValueError('Figure was already styled with different scaling factors')
        return
    if legacy_text_scale != 1:
        for text in fig.findobj(mpl.text.Text):
            text.set_fontsize(text.get_fontsize() * legacy_text_scale)
    if legacy_line_scale != 1:
        for line in fig.findobj(Line2D):
            line.set_linewidth(line.get_linewidth() * legacy_line_scale)
    for ax in fig.axes:
        if ax.axison:
            frame_axes(ax, minor=False)
        if legacy_line_scale != 1:
            for collection in ax.collections:
                collection.set_linewidth(collection.get_linewidth() * legacy_line_scale)
    fig._hybrid_plotting_scales = scales


def validate_layout(fig):
    """Reject clipped text or figure legends overlapping visible axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    undrawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    undrawn.update((id(tick.label1), id(tick.label2)))
    for text in fig.findobj(mpl.text.Text):
        if id(text) in undrawn or not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if np.isfinite(box.extents).all() and (
            box.x0 < -1 or box.y0 < -1 or
            box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1
        ):
            raise ValueError(f'Text outside figure: {text.get_text()}')
    for legend in fig.legends:
        if any(legend.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer))
               for ax in fig.axes if ax.axison):
            raise ValueError('Figure legend intersects axes')


def save_figure(fig, stem, *, formats=('pdf', 'svg', 'png'), check_layout=True):
    """Export without changing sizes, limits, data, or statistics; no implicit close."""
    stem = Path(stem)
    formats = tuple(formats)
    if not formats or any(ext not in ('pdf', 'svg', 'png') for ext in formats):
        raise ValueError('formats must contain pdf, svg, and/or png')
    if check_layout:
        validate_layout(fig)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with mpl.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'path'}):
        for ext in formats:
            path = stem.parent / f'{stem.name}.{ext}'
            fig.savefig(path, dpi=300, facecolor='white')
            paths.append(path)
    return tuple(paths)
