import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
import copy
from tqdm import tqdm

from utils import calculate_eer


class ClusteringTester:
    def __init__(self, 
            embeddings, 
            labels,
            n_users_type1=5,
            n_utts_per_user_type1=5,
            n_users_type2=0,
            n_utts_per_user_type2=0,
            n_verify_user_utts=10,
            n_verify_guest_utts=50,
            n_tests=500):

        assert len(embeddings) == len(labels) > 0
        self.n = len(embeddings)
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)
        self.speakers = np.unique(labels)
        self.n_tests = n_tests

        np.random.seed(42)
        type2 = (n_users_type2 > 0)
        assert (n_users_type1 > 0 and n_utts_per_user_type1 > 0)
        if type2:
            assert n_utts_per_user_type2 > 0
        assert max(n_utts_per_user_type1, n_utts_per_user_type2) < 21 #voxceleb2

        self.enroll_indices = []
        self.enroll_labels = []
        self.verify_indices = []
        self.verify_labels = []
        self.guest_indices = []

        for _ in range(n_tests):
            users = np.random.choice(self.speakers, size=n_users_type1+n_users_type2, replace=False)
            if type2:
                users_type1 = users[:n_users_type1]
                users_type2 = users[n_users_type1:]
            else:
                users_type1 = users

            enroll_indices = []
            enroll_labels = []
            verify_indices = []
            verify_labels = []

            guest_mask = np.ones(self.n, dtype=bool)
            
            def add_user(user, n_utts):
                user_mask = (self.labels == user)
                guest_mask[user_mask] = False
                user_indices = np.random.permutation(np.arange(0, self.n)[user_mask])
                assert len(user_indices) > n_utts
                enroll_indices.extend(user_indices[:n_utts])
                enroll_labels.extend([user] * n_utts)
                n_ver = min(n_verify_user_utts, len(user_indices) - n_utts)
                verify_indices.extend(user_indices[-n_ver:])
                verify_labels.extend([user] * n_ver)

            for user in users_type1:
                add_user(user, n_utts_per_user_type1)
            if type2:
                for user in users_type2:
                    add_user(user, n_utts_per_user_type2)

            self.enroll_indices.append(enroll_indices)
            self.enroll_labels.append(enroll_labels)
            self.verify_indices.append(verify_indices)
            self.verify_labels.append(verify_labels)
            self.guest_indices.append(np.random.choice(np.arange(0, self.n)[guest_mask], size=n_verify_guest_utts, replace=False))


    def test(self, verificator_init, use_tqdm=False):

        correct_mask = np.zeros(self.n_tests, dtype=bool)
        v_measures = np.zeros(self.n_tests)
        homogeneities = np.zeros(self.n_tests)
        completenesses = np.zeros(self.n_tests)
        guest_detection_scores = []
        guest_detection_labels = []

        for i in (tqdm(range(self.n_tests)) if use_tqdm else range(self.n_tests)):
            verificator = copy.deepcopy(verificator_init)

            correct_enrollment = verificator.enroll(self.embeddings[self.enroll_indices[i]], self.enroll_labels[i], check_labels=True)
            correct_mask[i] = correct_enrollment

            verify_labels_pred, verify_user_scores = verificator.verify(self.embeddings[self.verify_indices[i]])
            _, verify_guest_scores = verificator.verify(self.embeddings[self.guest_indices[i]])

            homogeneities[i], completenesses[i], v_measures[i] = homogeneity_completeness_v_measure(self.verify_labels[i], verify_labels_pred)

            guest_detection_scores.extend(verify_user_scores + verify_guest_scores)
            guest_detection_labels.extend([True] * len(verify_user_scores) + [False] * len(verify_guest_scores))

        results = {
            'clustering_acc': correct_mask.mean(),
            'v_measure_total': v_measures.mean(),
            'v_measure_correct': v_measures[correct_mask].mean() if correct_mask.any() else 0.,
            'guest_detection_eer': calculate_eer(guest_detection_scores, guest_detection_labels)
        }

        return results
        
