import io
import os
import operator
import sys
import json

import pandas as pd
import numpy as np

from topmodel import hmetrics

THRESHOLD_BINS = 100
N_THRESHOLDS = 1. / THRESHOLD_BINS

SCORES_FILE = 'scores.tsv'
HISTOGRAM_FILE = 'histogram.json'
NOTES_FILE = "notes.txt"

BIN_COUNT = 100


class ModelDataManager(object):

    def __init__(self, file_system):
        self.file_system = file_system

    def list(self):
        models = []
        for path in self.file_system.list():
            filename = path.split('/')[-1]
            if filename in (SCORES_FILE):
                model_data = ModelData(self.file_system, path[:-len(filename)])
                models.append(model_data)

        return sorted(models, key=operator.attrgetter('model_path'))

    def search(self, target_model):
        return filter(lambda model: target_model in model.model_path, self.list())


class ModelData(object):

    def __init__(self, file_system, model_path):
        self.model_path = model_path
        self.file_system = file_system
        self.data_frame = None

    def get_metrics(self, bootstrap=False):
        def bootstrap_histogram(hist):
            def samp(x):
                return np.random.poisson(x)
            # i think this is right
            trues = [samp(x) for x in hist['trues']]
            falses = map(
                lambda x: samp(x[0] - x[1]), zip(hist['totals'], hist['trues']))
            return {'probs': hist['probs'],
                    'trues': trues,
                    'totals': map(lambda x: x[0] + x[1], zip(trues, falses))}

        def metrics_from_hist(hist):
            return {
                # facts about the histogram
                'thresholds': hist['probs'],
                'score_distribution': hist['totals'],
                'trues': hist['trues'],
                # 3 main metrics
                'precisions': hmetrics.precisions(hist),
                'recalls': hmetrics.recalls(hist),
                'fprs': hmetrics.fprs(hist),
                # extra metrics
                'marginal_precisions': hmetrics.marginal_precisions(hist),
                # single number metrics
                'brier': hmetrics.brier(hist),
                'logloss': hmetrics.logloss(hist)
            }

        hist = self.to_histogram_format()
        base = metrics_from_hist(hist)
        if bootstrap == False:
            return base
        else:
            return [base] + [metrics_from_hist(bootstrap_histogram(hist)) for x in range(bootstrap)]

    def to_data_frame(self, **kwargs):
        if self.data_frame is None:
            scores_path = os.path.join(self.model_path, SCORES_FILE)
            csv = self.file_system.read_file(scores_path)
            with io.BytesIO(csv) as f:
                self.data_frame = pd.read_csv(f, sep='\t', **kwargs)
            self.data_frame = self.data_frame.dropna(how='any')

        # what we do here is build the histogram of the data, quantized to 100 bins
        # that's a O(1) size representation

        # alternate data format is "score,trues,falses"
        # here we build the DataFrame to match the old scores.tsv
        # sadly this is slow...

        # TODO: needlessly slow
        raw = self.data_frame
        if 'trues' in raw.columns:
            actual = []
            pred_score = []
            # not very pythonic...
            for i in range(len(raw)):
                tr = raw.irow(i)
                pred_score += [tr.score] * int(tr.trues + tr.falses)
                actual += [False] * int(tr.falses)
                actual += [True] * int(tr.trues)
            self.data_frame = pd.DataFrame(
                data={'actual': actual, 'pred_score': pred_score})

        return self.data_frame

    def to_histogram_format(self):
        histogram_path = os.path.join(self.model_path, HISTOGRAM_FILE)
        histogram_json = self.file_system.read_file(histogram_path)
        if histogram_json != None:
            ret = json.loads(histogram_json)
        else:
            df = self.to_data_frame()
            actual = df.get('actual')
            predicted = df.get('pred_score')
            bin_edges = map(
                lambda x: x * 1.0 / THRESHOLD_BINS, range(0, THRESHOLD_BINS + 1))
            o_count = []
            f_count = []

            # actual count in each of the bins
            probs = []

            for i in range(THRESHOLD_BINS):
                probs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                o_count.append(np.count_nonzero(
                               ((predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])) & actual))
                f_count.append(np.count_nonzero(
                               ((predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1]))))
            # the probabilities, the number of true, and the total number
            ret = {'probs': probs, 'trues': o_count, 'totals': f_count}

            # cache it
            self.file_system.write_file(histogram_path, json.dumps(ret))

        return ret

    def save_data_frame(self, df):
        self.data_frame = df
        with io.BytesIO() as f:
            df.to_csv(f, sep='\t', index=False)
            scores_path = os.path.join(self.model_path, SCORES_FILE)
            self.file_system.write_file(scores_path, f.getvalue())

    def get_notes(self):
        notes_path = os.path.join(self.model_path, NOTES_FILE)
        return self.file_system.read_file(notes_path)

    def set_notes(self, note):
        notes_path = os.path.join(self.model_path, NOTES_FILE)
        return self.file_system.write_file(notes_path, note)
