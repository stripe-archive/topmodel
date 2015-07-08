import io
import os
import operator
import sys
import json
import time

import pandas as pd
import numpy as np

from topmodel import hmetrics

THRESHOLD_BINS = 100
N_THRESHOLDS = 1. / THRESHOLD_BINS

SCORES_FILE = 'scores.tsv'
ACTUALS_FILE = 'actuals.tsv'
SCORES_BM_FILE = 'scores_bm.tsv'

HISTOGRAM_FILE = 'histogram.json'
NOTES_FILE = "notes.txt"

BIN_COUNT = 100


class ModelDataManager(object):

    def __init__(self, file_system):
        self.file_system = file_system
        self.models = {}

        scores = [fp for fp in self.file_system.list()
                  if fp.endswith(SCORES_FILE)]
        for filepath in scores:
            basedir, _ = os.path.split(filepath)
            model_data = ModelData(self.file_system, basedir)
            self.models[basedir] = model_data

        actuals = [fp for fp in self.file_system.list()
                   if fp.endswith(ACTUALS_FILE)]
        for path in actuals:
            basedir = os.path.split(path)[0]
            scores_bm = [p for p in self.file_system.list(basedir)
                         if p.endswith(SCORES_BM_FILE)]
            for fp in scores_bm:
                model_path, _ = os.path.split(fp)
                model_data = BenchmarkedModelData(self.file_system, model_path)
                self.models[model_path] = model_data

    def search(self, target_model):
        return filter(lambda model: target_model in model.model_path, self.list())


class ModelData(object):

    def __init__(self, file_system, model_path):
        self.model_path = model_path
        self.file_system = file_system
        self.data_frame = None

    def get_created_time(self):
        return time.ctime(os.path.getctime(self.model_path))

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
            bootstrapped = [
                metrics_from_hist(bootstrap_histogram(hist))
                for x in range(bootstrap)]
            return [base] + bootstrapped

    def check_alt_format(self):
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

    def to_data_frame(self, **kwargs):
        if self.data_frame is None:
            scores_path = os.path.join(self.model_path, SCORES_FILE)
            csv = self.file_system.read_file(scores_path)
            with io.BytesIO(csv) as f:
                self.data_frame = pd.read_csv(f, sep='\t', **kwargs)
            self.data_frame = self.data_frame.dropna(how='any')

        self.check_alt_format()
        return self.data_frame

    def to_histogram_format(self):
        # Build histogram of the data quantized to THRESHOLD_BINS bins
        # that's a O(1) size representation
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


class BenchmarkedModelData(ModelData):
    """
    Benchmarked model data. Store actuals in a separate 'actual.tsv' file
    with observations identifiers in an 'id' column and actuals in an
    'actual' column. Upload scores in a 'scores_bm.tsv' file that has
    a matching 'id' column' and 'pred_scores' column. Throws an error
    if the scores and actuals do not completely line up.
    """
    def indexed_data_frame(self, path, **kwargs):
        raw = self.file_system.read_file(path)
        with io.BytesIO(raw) as f:
          df = pd.read_csv(f, sep='\t', index_col=False, **kwargs)
        assert df.duplicated('id').sum() == 0, "id column is not unique"
        return df.set_index('id')

    def to_data_frame(self, **kwargs):
        if self.data_frame is None:
            basedir, _ = os.path.split(self.model_path)
            actuals_path = os.path.join(basedir, ACTUALS_FILE)
            df_actuals = self.indexed_data_frame(actuals_path, **kwargs)
            scores_path = os.path.join(self.model_path, SCORES_BM_FILE)
            df_scores = self.indexed_data_frame(scores_path, **kwargs)
            assert sorted(df_actuals.index) == sorted(df_scores.index), \
                "Indices for actuals and scores do not match"

            self.data_frame = pd.merge(
                df_actuals, df_scores, left_index=True, right_index=True)
            self.data_frame = self.data_frame.dropna(how='any')

        self.check_alt_format()
        return self.data_frame

    def get_created_time(self):
        return time.ctime(os.path.getctime(self.model_path))
