from math import floor
from cStringIO import StringIO

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpld3 import plugins, save_html


def make_points_far(xs, ys, thresholds, min_dist=0.03):
    """
    Return a subset of points and thresholds such that the ys are at least
    `min_dist` apart. 0.03 seems to be a readable value. Always includes the
    first and last points in xs and ys.
    """
    y_prev = None
    new_xs = []
    new_ys = []
    new_thresholds = []
    for x, y, threshold in zip(xs[:-1], ys[:-1], thresholds[:-1]):
        if y_prev is None or (np.abs(y - y_prev) >= min_dist):
            y_prev = y
            new_xs.append(x)
            new_ys.append(y)
            new_thresholds.append(threshold)
    # Always keep the last point (xs[-1], ys[-1])
    new_xs.append(xs[-1])
    new_ys.append(ys[-1])
    new_thresholds.append(thresholds[-1])
    return (new_xs, new_ys, new_thresholds)


def draw_labels(ax, xs, ys, thresholds, labels_left=False):
    font = matplotlib.font_manager.FontProperties(family='Tahoma', size=6)
    for x, y, threshold in zip(xs, ys, thresholds):
        x_round = floor(x * 1000) / 1000
        y_round = floor(y * 1000) / 1000
        threshold_round = floor(threshold * 1000) / 1000
        coords = (x + 0.01, y)
        if labels_left:
            coords = (x - 0.190, y - 0.01)
        annotation = "{threshold}: [{x},{y}] ".format(x=x_round, y=y_round,
                                                      threshold=threshold_round)
        ax.annotate(annotation, coords, fontproperties=font)


def plot_boxplot(vals, label):
    fig, ax = plt.subplots()
    ax.boxplot(vals)
    plt.setp(ax, xticklabels=label)
    return save_image()


def plot_scatter(x, y, xlabel, ylabel, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, marker="o", color="purple")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    plt.tight_layout()
    return save_image()


def plot_xy(xs, ys, thresholds, xlabel, ylabel, labels=True, labels_left=False,
            ax=None, xlim=(0, 1), ylim=(0, 1), autofmt_xdate=False, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        if autofmt_xdate:
            fig.autofmt_xdate()

    ax.plot(xs, ys, '-o', **plot_kwargs)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        draw_labels(ax, xs, ys, thresholds, labels_left=labels_left)
    plt.tight_layout()
    return save_image()


def pretty_point(coord):
    return "0.0" if (coord is None) else ("%.2f" % coord)


def plot_xy_bootstrapped(xs, ys, thresholds, xlabel, ylabel, labels=False, labels_left=False, ax=None, label=None, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(1, len(xs)):
        ax.plot(xs[i], ys[i], '-', alpha=0.3)
    (xs_, ys_, thresholds_) = make_points_far(xs[0], ys[0], thresholds)
    label_text = ["Threshold: %s (%s, %s)" % (t, pretty_point(x), pretty_point(y)) for (x, y, t) in zip(xs_, ys_, thresholds_)]
    if label is None:
        scatter = ax.plot(xs_, ys_, '-o', **plot_kwargs)
        plugins.connect(fig, plugins.PointHTMLTooltip(scatter[0], label_text))
    else:
        scatter = ax.plot(xs_, ys_, '-o', label=label, **plot_kwargs)
        plugins.connect(fig, plugins.PointHTMLTooltip(scatter[0], label_text))
    if labels:
        draw_labels(ax, xs_, ys_, thresholds_, labels_left=labels_left)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if label is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='best')
    return save_image()


def plot_scores_histogram_log(thresholds, all_counts, true_counts, xlabel, ax=None):
    plt.figure()
    # First graph
    if ax is None:
        _, ax = plt.subplots()
    falses = [i - j for i, j in zip(all_counts, true_counts)]
    width = (thresholds[1] - thresholds[0])/2
    offset = [i + width for i in thresholds]
    plt.bar(offset, falses, width=width,
            log=True, label="False items")
    plt.bar(thresholds, true_counts, width=width,
            log=True, color="purple", label="True items")
    plt.grid(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    # Write to image
    image_data = StringIO()
    plt.xlim((0.0, 1.0))
    plt.xlabel(xlabel)
    plt.legend(loc='best')
    plt.savefig(image_data, format='svg')
    image_data.seek(0)
    return image_data


def plot_absolute_score_histogram(thresholds, all_counts, true_counts, xlabel, ax=None):
    plt.figure()
    # First graph
    if ax is None:
        _, ax = plt.subplots()
    falses = [i - j for i, j in zip(all_counts, true_counts)]
    width = (thresholds[1] - thresholds[0])/2
    offset = [i + width for i in thresholds]
    plt.bar(offset, falses , width=width,
            log=False, label="False items")
    plt.bar(thresholds, true_counts, width=width,
            log=False, color="purple", label="True items")
    plt.grid(False)
    # Write to image
    image_data = StringIO()
    plt.xlim((0.0, 1.0))
    plt.xlabel(xlabel)
    plt.legend(loc='best')
    plt.savefig(image_data, format='svg')
    image_data.seek(0)
    return image_data


def save_image(ax=None):
    image_data = StringIO()
    save_html(plt.gcf(), image_data)
    image_data.seek(0)
    return image_data
