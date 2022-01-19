import numpy as np
import pickle
from tqdm import tqdm

from utils import calculate_eer, cosine


EMBEDDINGS = 'data/embeddings_labeled.pkl'
OUTPUT = 'data/cos_matrix.pkl'


with open(EMBEDDINGS, 'rb') as f:
    embeddings, labels = pickle.load(f)

assert len(embeddings) == len(labels)
n = len(embeddings)

cos_matrix = np.zeros((n, n))

for i in tqdm(range(n)):
    for j in range(n):
        cos_matrix[i, j] = cosine(embeddings[i], embeddings[j])

with open(OUTPUT, 'wb') as f:
    pickle.dump(cos_matrix, f)
