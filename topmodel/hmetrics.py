import numpy as np


def recalls(hist):
    # true positive rate
    # TP / (TP + FN)
    ret = []
    trues = sum(hist['trues'])
    all_trues = trues
    for _, true in zip(hist['thresholds'], hist['trues']):
        ret.append(trues * 1.0 / all_trues if all_trues != 0 else None)
        trues -= true
    return ret


def fprs(hist):
    # FP / (FP + TN)
    # thresholds is being used as the threshold
    # ones selected that aren't true / all selected
    ret = []
    falses = sum(hist['totals']) - sum(hist['trues'])
    all_falses = falses
    for _, true, total in zip(hist['thresholds'], hist['trues'], hist['totals']):
        ret.append(falses * 1.0 / all_falses if all_falses != 0 else None)
        falses -= (total - true)
    return ret


def precisions(hist):
    ret = []
    selected = sum(hist['totals'])
    trues = sum(hist['trues'])
    for _, true, total in zip(hist['thresholds'], hist['trues'], hist['totals']):
        ret.append(trues * 1.0 / selected if selected != 0 else None)
        trues -= true
        selected -= total
    return ret


def marginal_precisions(hist):
    return map(lambda x: x[0] * 1.0 / x[1] if x[1] != 0 else None, zip(hist['trues'], hist['totals']))


def logloss(hist):
    loss = 0.0
    N = sum(hist['totals'])
    for threshold, true, total in zip(hist['thresholds'], hist['trues'], hist['totals']):
        t = true
        f = total - t
        loss += t * \
            np.log(threshold) + f * np.log(1.0 - threshold)
    return -loss / N


def auc(fprs, tprs):
    xs = np.concatenate(([0], fprs[::-1], [1]))
    ys = np.concatenate(([0], tprs[::-1], [1]))
    return np.trapz(ys, xs)
