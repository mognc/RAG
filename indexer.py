import numpy as np
import faiss

# Load your index (adjust according to your setup)
index = faiss.read_index('data/faiss_index_HNSWFLAT.index')

def search_passages(query_embedding, index, k=5):
    D, I = index.search(query_embedding, k)  # Search for k nearest neighbors
    return I[0]  # Return the indices of the closest passages
