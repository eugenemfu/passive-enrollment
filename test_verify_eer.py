import numpy as np
import pickle
from tqdm import tqdm

from utils import calculate_eer, cosine


EMBEDDINGS = 'data/embeddings_labeled.pkl'


with open(EMBEDDINGS, 'rb') as f:
	embeddings, labels = pickle.load(f)

assert len(embeddings) == len(labels)
n = len(embeddings)

scores = []
bools = []

for i in tqdm(range(n)):
    for j in range(i + 1, n):
        scores.append(cosine(embeddings[i], embeddings[j]))
        bools.append(labels[i] == labels[j])

eer = calculate_eer(scores, bools)
print(eer)
