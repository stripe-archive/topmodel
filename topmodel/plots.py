from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from topmodel import plot_helpers


def _metrics_table(cached_data):
    count = np.array(cached_data['score_distribution'])
    total = count.sum()
    table = pd.DataFrame({
                         'Precision': cached_data['precisions'],
                         'Recall/TPR': cached_data['recalls'],
                         'FPR': cached_data['fprs'],
                         'Threshold': cached_data['thresholds'],
                         'N Predicted': np.append([total], (total - count.cumsum())[:-1])})
    table = table.set_index('Threshold')
    return table


def utf8_decode(image_data):
    return image_data.read().decode('utf-8')


def precision_recall_curve(cached_data, ax=None, label=None):
    thresholds = cached_data[0]['thresholds']
    precision = [x['precisions'] for x in cached_data]
    recall = [x['recalls'] for x in cached_data]

    image_data = plot_helpers.plot_xy_bootstrapped(
        precision, recall, thresholds, 'precision', 'recall', ax=ax, label=label)
    return utf8_decode(image_data)


def roc_curve(cached_data, ax=None, label=None):
    thresholds = cached_data[0]['thresholds']
    fpr = [x['fprs'] for x in cached_data]
    tpr = [x['recalls'] for x in cached_data]

    image_data = plot_helpers.plot_xy_bootstrapped(
        fpr, tpr, thresholds, 'false positive', 'true positive', ax=ax, label=label)
    return utf8_decode(image_data)


def marginal_precision_curve(cached_data):
    image_data = plot_helpers.plot_scatter(
        cached_data['thresholds'], cached_data['marginal_precisions'], 'predicted', 'actual')
    return utf8_decode(image_data)


def thresholds_graph(cached_data):
    table = _metrics_table(cached_data)
    table.plot(secondary_y='N Predicted')
    image_data = plot_helpers.save_image()
    return utf8_decode(image_data)


def thresholds_table(cached_data):
    table = _metrics_table(cached_data)
    html = table.to_html()
    return html.replace('class="dataframe"', 'class="table table-striped table-bordered table-condensed"')


def score_distribution(cached_data):
    image_data = plot_helpers.plot_scores_histogram_log(
        cached_data['thresholds'], cached_data['score_distribution'],
        cached_data['trues'],
        'Score')
    return utf8_decode(image_data)

def absolute_score_distribution(cached_data):
    image_data = plot_helpers.plot_absolute_score_histogram(
        cached_data['thresholds'], cached_data['score_distribution'],
        cached_data['trues'],
        'Score')
    return utf8_decode(image_data)
