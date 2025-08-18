import numpy as np

clusters = np.load("exp/clusters.npy")
embeddings = np.load("exp/embeddings.npy")

print(clusters.shape)
print(embeddings.shape)