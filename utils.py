import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def list_average(a):
    return sum(a) / len(a)


def cosine(emb1, emb2):
    return (emb1 @ emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def calculate_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=True)
    tpr_curve = interp1d(fpr, tpr)
    # thr_curve = interp1d(fpr, thresholds)
    eer = brentq(lambda x: 1. - x - tpr_curve(x), 0., 1.)
    # frfa1 = 1 - tpr_curve(0.01)
    # frfa01 = 1 - tpr_curve(0.001)
    # frfa001 = 1 - tpr_curve(0.0001)
    # thr_eer = thr_curve(eer)
    # thr1 = thr_curve(0.01)
    # thr001 = thr_curve(0.0001)
    return eer