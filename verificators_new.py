import random
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, MeanShift, AffinityPropagation, DBSCAN, SpectralClustering
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import pandas as pd
import copy

from utils import cosine, list_average, get_cosine_matrix


class Verificator:
    def __init__(self, preenroll_threshold=None, preenroll_method='remove'):
        self.enrolled = False
        self.speaker_embeddings = []
        self.preenroll_method = preenroll_method
        self.preenroll_threshold = preenroll_threshold

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
        for label in sorted(list(speaker_embeddings.keys())):
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
        
    def preenroll(self, avg_embeddings):
        self.speaker_embeddings = copy.deepcopy(avg_embeddings)
        #print(avg_embeddings)
        if self.preenroll_method == 'remove' and len(avg_embeddings) > 0:
            if self.preenroll_threshold is None:
                self.speaker_embeddings = []

    def enroll(self, embeddings, labels, check_labels=False, show_tsne=False):
        self.check_enroll(embeddings, labels)
        #print(labels)
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        mask = np.ones(len(embeddings), dtype=bool)
        labels_pred = - np.ones(len(embeddings))
        if self.preenroll_method == 'remove':
            #print(embeddings)
            #print(self.preenroll_threshold)
            labels_pred = np.array(self.verify(embeddings, threshold=self.preenroll_threshold)[0])
        mask = (labels_pred == -1)
        #print(len(self.speaker_embeddings))
        #print(labels_pred)
        if mask.sum() < 2:
            labels_pred_mask = np.zeros(mask.sum())
        else:
            labels_pred_mask = self.enroll_predict_labels(embeddings[mask], labels[mask])
        for i in range(len(labels_pred_mask)):
            if labels_pred_mask[i] != -1:
                labels_pred_mask[i] += len(self.speaker_embeddings)
        #print(labels_pred)
        labels_pred[mask] = labels_pred_mask
        #print(labels_pred)
        #print(labels)
        self.enroll_labeled_embeddings(embeddings[mask], labels_pred_mask)
        #print(labels_pred)
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
            threshold = 2
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
            labels.append(best_speaker if 1 - best_score < threshold else -1)
            scores.append(best_score)
        return labels, scores


class ActiveVerificator(Verificator):
    def enroll_predict_labels(self, embeddings, labels):
        return np.unique(labels, return_inverse=True)[1]


class AgglomerativeVerificator(Verificator):
    def __init__(self, clustering_threshold, affinity='cosine', linkage='average', min_cluster_size=1, **kwargs):
        super().__init__(**kwargs)
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
            #print(embeddings.shape)
            labels_pred = model.fit_predict(embeddings)
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
                if 1 - max_dist > self.clustering_threshold:
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

            
class AgglomerativeUpdateVerificator(Verificator):
    def __init__(self, clustering_threshold, min_cluster_size=1, **kwargs):
        super().__init__(**kwargs)
        self.clustering_threshold = clustering_threshold
        self.min_cluster_size = min_cluster_size
        assert self.preenroll_method == 'update'
        
    def enroll_predict_labels(self, embeddings, labels):
        aff_mat = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(i, len(embeddings)):
                cos = cosine(embeddings[i], embeddings[j])
                aff_mat[i, j] = cos
                aff_mat[j, i] = cos
        labels_pred, _ = self.verify(embeddings, threshold=self.preenroll_threshold)
        #print(labels_pred)
        self.speaker_embeddings = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if labels_pred[i] == labels_pred[j] != -1:
                    aff_mat[i, j] = 1
                    aff_mat[j, i] = 1
                    #print(1)
                if -1 != labels_pred[i] != labels_pred[j] != -1:
                    aff_mat[i, j] = 0
                    aff_mat[j, i] = 0
                    #print(0)
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity='precomputed',
            linkage='average',
            distance_threshold=self.clustering_threshold)
        labels_pred = model.fit_predict(1 - aff_mat)
        #print(labels_pred)
        labels_unique, labels_counts = np.unique(labels_pred, return_counts=True)
        labels_counts = dict(zip(labels_unique, labels_counts))
        for i, label in enumerate(labels_pred):
            if labels_counts[label] < self.min_cluster_size:
                labels_pred[i] = -1
        #print(labels_pred)
        return labels_pred
    

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