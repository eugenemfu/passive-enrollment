import numpy as np
import pickle
from tqdm import tqdm
import json

from utils import calculate_eer, cosine


EMBEDDINGS = 'data/embeddings_labeled.pkl'


with open(EMBEDDINGS, 'rb') as f:
	embeddings, labels = tuple(map(np.array, pickle.load(f)))

assert len(embeddings) == len(labels)
n = len(embeddings)
speakers = np.unique(labels)

results = []

np.random.seed(42)

for k in range(1, 10):
    scores = []
    bools = []

    for speaker in tqdm(speakers):
        speaker_mask = labels == speaker
        for _ in range(200):
            speaker_ids = np.random.choice(np.arange(n)[speaker_mask], k+10, replace=False)
            speaker_embedding = embeddings[speaker_ids[:k]].mean(0)
            for i in speaker_ids[k:]:
                scores.append(cosine(speaker_embedding, embeddings[i]))
                bools.append(True)
            for i in np.random.choice(np.arange(n)[~speaker_mask], 10, replace=False):
                scores.append(cosine(speaker_embedding, embeddings[i]))
                bools.append(False)

    eer = calculate_eer(scores, bools)
    print(k, eer)
    results.append((k, eer))

json.dump(results, open('data/eer_results.json', 'w'))
