import random
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, MeanShift, AffinityPropagation, DBSCAN, SpectralClustering
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import pandas as pd

from utils import cosine, list_average, get_cosine_matrix


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
            if label != -1:
                speaker_embeddings[label].append(embedding)
        self.speaker_labels = list(speaker_embeddings.keys())
        for label in self.speaker_labels:
            self.speaker_embeddings.append(list_average(speaker_embeddings[label]))

    def check_labels(self, labels_true, labels_pred):
        np_labels_pred = np.array(labels_pred)
        np_labels_true = np.array(labels_true)
        labels_true_ = np_labels_true[np_labels_pred != -1]
        labels_pred_ = np_labels_pred[np_labels_pred != -1]
        #print(np_labels_pred != -1)
        #print(labels_true_)
        self.speakers_labels_enrolled = np.unique(labels_true_)
        self.part_speakers_found = len(self.speakers_labels_enrolled) / len(np.unique(labels_true))
        true_n_clusters = len(np.unique(labels_true_))
        pred_n_clusters = len(np.unique(labels_pred_))
        equal_clusters = true_n_clusters == pred_n_clusters
        ideal_clusters = v_measure_score(labels_true_, labels_pred_) > 0.99
        more_clusters = pred_n_clusters > true_n_clusters
        less_clusters = pred_n_clusters < true_n_clusters
        if equal_clusters and ideal_clusters:
            return 0
        elif equal_clusters:
            return 1
        elif less_clusters:
            return 2
        elif more_clusters:
            return 3
        else:
            assert False
            
    def show_tsne(self, embeddings, labels_true, labels_pred):
        tsne = TSNE(n_components=2, metric='cosine', square_distances=True, init='random', learning_rate='auto', perplexity=10)
        emb_tr = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(emb_tr[:, 0], emb_tr[:, 1], c=pd.factorize(labels_true)[0])
        plt.subplot(1, 2, 2)
        plt.scatter(emb_tr[:, 0], emb_tr[:, 1], c=pd.factorize(labels_pred)[0])
        plt.show()

    def enroll(self, embeddings, labels, check_labels=False, show_tsne=False):
        self.check_enroll(embeddings, labels)
        #print(labels)
        labels_pred = self.enroll_predict_labels(embeddings, labels)
        #print(labels_pred)
        self.enroll_labeled_embeddings(embeddings, labels_pred)
        if show_tsne:
            self.show_tsne(embeddings, labels, labels_pred)
        if check_labels:
            res = self.check_labels(labels, labels_pred)
            if show_tsne:
                if res > 0:
                    print(labels, labels_pred)
                print(res)
            return res

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
        return np.unique(labels, return_inverse=True)[1]


class AgglomerativeVerificator(Verificator):
    def __init__(self, clustering_threshold, affinity, linkage='average', min_cluster_size=1):
        super().__init__()
        self.clustering_threshold = clustering_threshold
        self.affinity = affinity
        self.linkage = linkage
        self.min_cluster_size = min_cluster_size

    def enroll_predict_labels(self, embeddings, labels):
        if self.linkage != 'center':
            model = AgglomerativeClustering(
                n_clusters=None,
                affinity=self.affinity,
                linkage=self.linkage,
                distance_threshold=self.clustering_threshold)
            labels_pred = model.fit_predict(embeddings, labels)
            labels_unique, labels_counts = np.unique(labels_pred, return_counts=True)
            labels_counts = dict(zip(labels_unique, labels_counts))
            for i, label in enumerate(labels_pred):
                if labels_counts[label] < self.min_cluster_size:
                    labels_pred[i] = -1
            return labels_pred
        elif self.affinity == 'cosine':
            clusters = [[embeddings[i], [i]] for i in range(len(embeddings))]
            while len(clusters) > 1:
                max_dist = -1
                argmax = None
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        dist = cosine(clusters[i][0], clusters[j][0])
                        if dist > max_dist:
                            max_dist = dist
                            argmax = (i, j)
                #print(argmax, max_dist)
                if max_dist < self.clustering_threshold:
                    #print('stopped')
                    break
                i, j = argmax
                n_i = len(clusters[i][1])
                n_j = len(clusters[j][1])
                clusters[i][0] = (n_i * clusters[i][0] + n_j * clusters[j][0]) / (n_i + n_j)
                clusters[i][1].extend(clusters[j][1])
                clusters.pop(j)
                #print([cluster[1] for cluster in clusters])
            result = - np.ones(len(embeddings))
            for i in range(len(clusters)):
                if len(clusters[i][1]) < self.min_cluster_size:
                    continue
                for j in clusters[i][1]:
                    result[j] = i
            #print(result)
            return result
        else:
            raise ValueError('center works with cosine only')


class MeanShiftVerificator(Verificator):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth

    def enroll_predict_labels(self, embeddings, labels):
        model = MeanShift(bandwidth=self.bandwidth, cluster_all=False)
        return model.fit_predict(embeddings, labels)


class AffinityVerificator(Verificator):
    def __init__(self, affinity='cosine', damping=0.95):
        super().__init__()
        self.affinity = affinity
        self.damping = damping

    def enroll_predict_labels(self, embeddings, labels):
        X = embeddings
        if self.affinity == 'cosine':
            self.affinity = 'precomputed'
            X = get_cosine_matrix(embeddings)
        model = AffinityPropagation(damping=self.damping, affinity=self.affinity, random_state=0, max_iter=400)
        return model.fit_predict(X, labels)


class DBSCANVerificator(Verificator):
    def __init__(self, eps, min_samples=3):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def enroll_predict_labels(self, embeddings, labels):
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        return model.fit_predict(embeddings, labels)
    

class SpectralVerificator(Verificator):
    def __init__(self):
        super().__init__()
        
    def enroll_predict_labels(self, embeddings, labels):
        X = get_cosine_matrix(embeddings)
        #X = (X + 1) / 2
        #X = X @ X.T
        # print(X)
        eigval, _ = np.linalg.eig(X)
        eigval = np.flip(np.sort(eigval)) + 1e-13
        # [print(val) for val in eigval]
        # print(eigvec)
        assert eigval[-1] > 0, eigval
        n_clusters = np.argmax([eigval[i] / eigval[i + 1] for i in range(min(15, len(eigval) - 1))]) + 1
        #print(n_clusters)
        model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        result = model.fit_predict((X + 1) / 2, labels)
        #print(labels)
        #print(result)
        return result