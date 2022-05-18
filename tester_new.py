import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure, fowlkes_mallows_score
import copy
from tqdm import tqdm
import pandas as pd

from utils import calculate_eer


class ClusteringTester:
    def __init__(self, 
            embeddings, 
            labels,
            n_users_enrolled=0,
            n_utts_per_user_enrolled=10,
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
        enrolled = (n_users_enrolled * n_utts_per_user_enrolled > 0)
        type1 = (n_users_type1 * n_utts_per_user_type1 > 0)
        type2 = (n_users_type2 * n_utts_per_user_type2 > 0)
        assert enrolled or type1 or type2
        # assert max(n_utts_per_user_type1, n_utts_per_user_type2) < 21 #voxceleb2
        
        self.enrolled_embeddings = []
        self.enroll_indices = []
        self.enroll_labels = []
        self.verify_indices = []
        self.verify_labels = []
        self.guest_indices = []

        for _ in range(n_tests):
            users = np.random.choice(self.speakers, size=n_users_enrolled+n_users_type1+n_users_type2, replace=False)
            users_enrolled = users[:n_users_enrolled]
            users_type1 = users[n_users_enrolled:n_users_enrolled+n_users_type1]
            users_type2 = users[n_users_enrolled+n_users_type1:]

            enrolled_embeddings = []
            enrolled_labels = []
            enroll_indices = []
            enroll_labels = []
            verify_indices = []
            verify_labels = []

            guest_mask = np.ones(self.n, dtype=bool)
            
            def add_user(user, n_utts, preenrolled=False):
                n_pre = 5 if preenrolled else 0
                user_mask = (self.labels == user)
                guest_mask[user_mask] = False
                user_indices = np.random.permutation(np.arange(0, self.n)[user_mask])
                assert len(user_indices) > n_utts + n_pre
                preenrolled_indices = user_indices[:n_pre]
                if preenrolled:
                    # print(preenrolled_indices)
                    # print(self.embeddings[preenrolled_indices])
                    # print(sum(self.embeddings[preenrolled_indices]))
                    enrolled_embeddings.append(sum(self.embeddings[preenrolled_indices]))
                    enrolled_labels.append(user)
                enroll_indices.extend(user_indices[n_pre:n_pre+n_utts])
                enroll_labels.extend([user] * n_utts)
                n_ver = min(n_verify_user_utts, len(user_indices) - n_utts - n_pre)
                verify_indices.extend(user_indices[-n_ver:])
                verify_labels.extend([user] * n_ver)
            
            for user in users_enrolled:
                add_user(user, n_utts_per_user_enrolled, preenrolled=True)
            for user in users_type1:
                add_user(user, n_utts_per_user_type1)
            for user in users_type2:
                add_user(user, n_utts_per_user_type2)
                
            self.enrolled_embeddings.append(enrolled_embeddings)
            self.enroll_indices.append(enroll_indices)
            self.enroll_labels.append(enroll_labels)
            self.verify_indices.append(verify_indices)
            self.verify_labels.append(np.array(verify_labels))
            self.guest_indices.append(np.random.choice(np.arange(0, self.n)[guest_mask], size=n_verify_guest_utts, replace=False))
        #print(len(self.enrolled_embeddings))


    def test(self, verificator_init, use_tqdm=False, show_tsne=False):
        correct_mask = np.zeros(self.n_tests, dtype=bool)
        equal_mask = np.zeros(self.n_tests, dtype=bool)
        less_mask = np.zeros(self.n_tests, dtype=bool)
        more_mask = np.zeros(self.n_tests, dtype=bool)
        v_measures = np.zeros(self.n_tests)
        homogeinities = np.zeros(self.n_tests)
        completenesses = np.zeros(self.n_tests)
        found_rates = np.zeros(self.n_tests)
        guest_detection_scores = [1]
        guest_detection_labels = [True]

        for i in (tqdm(range(self.n_tests)) if use_tqdm else range(self.n_tests)):
            verificator = copy.deepcopy(verificator_init)
            
            #print(i, len(self.enrolled_embeddings[i]))
            verificator.preenroll(self.enrolled_embeddings[i])
            enrollment_result = verificator.enroll(self.embeddings[self.enroll_indices[i]], self.enroll_labels[i], check_labels=True, show_tsne=show_tsne)
            correct_mask[i] = enrollment_result == 0
            equal_mask[i] = enrollment_result == 1
            less_mask[i] = enrollment_result == 2
            more_mask[i] = enrollment_result == 3
            
            mask_found = np.zeros(len(self.verify_labels[i]), dtype=bool)
            for j, label in enumerate(self.verify_labels[i]):
                mask_found[j] = label in verificator.speakers_labels_enrolled
                
            #print(len(verificator.speaker_embeddings))
            verify_labels_pred, verify_user_scores = verificator.verify(self.embeddings[np.array(self.verify_indices[i])[mask_found]])
            _, verify_guest_scores = verificator.verify(self.embeddings[self.guest_indices[i]])
            verify_labels_pred = np.array(verify_labels_pred)
            #print(verificator.preenroll_threshold)
            #print(verify_labels_pred)
            #print(mask_found)
            #print(self.verify_labels[i])

            homogeinities[i], completenesses[i], v_measures[i] = homogeneity_completeness_v_measure(self.verify_labels[i][mask_found], verify_labels_pred)
            #print(v_measures[i])

            guest_detection_scores.extend(verify_user_scores + verify_guest_scores)
            guest_detection_labels.extend([True] * len(verify_user_scores) + [False] * len(verify_guest_scores))
            
            found_rates[i] = verificator.part_speakers_found
            
        #print(v_measures)
        
        if found_rates.mean() > 0:
            results = {
                'v_measure': v_measures.mean(),
                'homogeinity': homogeinities.mean(),
                'completeness': completenesses.mean(),
                'guest_detection_eer': calculate_eer(guest_detection_scores, guest_detection_labels),
                'clustering_acc': correct_mask.mean(),
                'equal_rate': equal_mask.mean(),
                'less_rate': less_mask.mean(),
                'more_rate': more_mask.mean(),
                'found_rate': found_rates.mean()
            }
        else:
            results = {
                'found_rate': found_rates.mean()
            }

        return results
    
    def test_all(self, verificator_func, thresholds, use_tqdm=True):
        results = []
        for th in tqdm(thresholds) if use_tqdm else thresholds:
            row = {'threshold': th}
            row.update(self.test(verificator_func(th)))
            #print(row)
            results.append(row)
        results = pd.DataFrame(results)
        return results