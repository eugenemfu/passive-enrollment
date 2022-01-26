import random
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering, MeanShift, AffinityPropagation, DBSCAN
from sklearn.metrics import v_measure_score

from utils import cosine, list_average


class Verificator:
    def __init__(self):
        self.enrolled = False
        self.speaker_labels = []
        self.speaker_embeddings = []

    def check_enroll(self, embeddings, labels):
        if self.enrolled:
            raise RuntimeError('second enrollment')
        self.enrolled = True
        assert len(embeddings) == len(labels) > 0

    def enroll_predict_labels(self, embeddings, labels):
        pass

    def enroll_labeled_embeddings(self, embeddings, labels):
        speaker_embeddings = defaultdict(list)
        for embedding, label in zip(embeddings, labels):
            speaker_embeddings[label].append(embedding)
        self.speaker_labels = list(speaker_embeddings.keys())
        for label in self.speaker_labels:
            self.speaker_embeddings.append(list_average(speaker_embeddings[label]))

    def check_labels(self, labels_true, labels_pred):
        labels_true_ = labels_true[labels_pred != -1]
        labels_pred_ = labels_pred[labels_pred != -1]
        equal_len = (len(np.unique(labels_true)) == len(np.unique(labels_true_)) == len(np.unique(labels_pred_)))
        return equal_len and v_measure_score(labels_true_, labels_pred_) > 0.99

    def enroll(self, embeddings, labels, check_labels=False):
        self.check_enroll(embeddings, labels)
        labels_pred = self.enroll_predict_labels(embeddings, labels)
        self.enroll_labeled_embeddings(embeddings, labels_pred)
        if check_labels:
            return self.check_labels(labels, labels_pred)

    def verify(self, embeddings, threshold=None):
        if not self.enrolled:
            raise RuntimeError('verification before enrollment')
        if threshold is None:
            threshold = -1
        labels = []
        scores = []
        for embedding in embeddings:
            best_score = -1
            best_speaker = -1
            for i, speaker_embedding in enumerate(self.speaker_embeddings):
                score = cosine(embedding, speaker_embedding)
                if score > best_score:
                    best_score = score
                    best_speaker = i
            labels.append(self.speaker_labels[best_speaker] if best_score > threshold else 'GUEST')
            scores.append(best_score)
        return labels, scores


class ActiveVerificator(Verificator):
    def enroll_predict_labels(self, embeddings, labels):
        return labels


class AgglomerativeVerificator(Verificator):
    def __init__(self, clustering_threshold, affinity, linkage='average'):
        super().__init__()
        self.clustering_threshold = clustering_threshold
        self.affinity = affinity
        self.linkage = linkage

    def enroll_predict_labels(self, embeddings, labels):
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity=self.affinity,
            linkage=self.linkage,
            distance_threshold=self.clustering_threshold)
        return model.fit_predict(embeddings, labels)


class MeanShiftVerificator(Verificator):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth

    def enroll_predict_labels(self, embeddings, labels):
        model = MeanShift(bandwidth=self.bandwidth, cluster_all=False)
        return model.fit_predict(embeddings, labels)


class AffinityVerificator(Verificator):
    def __init__(self, affinity='cosine'):
        super().__init__()
        self.affinity = affinity

    def enroll_predict_labels(self, embeddings, labels):
        X = embeddings
        if self.affinity == 'cosine':
            self.affinity = 'precomputed'
            n = len(embeddings)
            X = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    cos = cosine(embeddings[i], embeddings[j])
                    X[i, j] = cos
                    X[j, i] = cos
        model = AffinityPropagation(damping=0.9, preference=0.77, affinity=self.affinity, random_state=0)
        return model.fit_predict(X, labels)
        

class DBSCANVerificator(Verificator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def enroll_predict_labels(self, embeddings, labels):
        model = DBSCAN(eps=self.eps, min_samples=3, metric='cosine')
        return model.fit_predict(embeddings, labels)

