import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from utils import evaluate_err


class ClusteringTester:
    def __init__(self, verificator, embeddings, labels):
        self.verificator = verificator
        self.embeddings = embeddings
        self.labels = labels

    def test(self,
             n_users_type1=5,
             n_utts_per_user_type1=5,
             n_users_type2=0,
             n_utts_per_user_type2=0,
             n_verify_user_utts=20,
             n_verify_guest_utts=20,
             n_tests=1000):
        np.random.seed(42)
        type2 = (n_users_type2 > 0)
        assert (n_users_type1 > 0 and n_utts_per_user_type1 > 0)
        if type2:
            assert n_utts_per_user_type2 > 0

        for i in range(n_tests):
            # sample family
            # enroll and verify utts
            # calculate metrics
