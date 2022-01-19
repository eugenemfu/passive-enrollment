import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from utils import cosine


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
        self.speaker_labels = speaker_embeddings.keys()
        for label in self.speaker_labels:
            self.speaker_embeddings.append(sum(speaker_embeddings[label]) / len(speaker_embeddings[label]))

    def enroll(self, embeddings, labels):
        self.enroll_check(embeddings, labels)
        labels_pred = self.enroll_predict_labels(embeddings, labels)
        self.enroll_labeled_embeddings(embeddings, labels_pred)

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



