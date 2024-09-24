import numpy as np
import faiss


index = faiss.read_index('data/faiss_index_HNSWFLAT.index')

def search_passages(query_embedding, index, k=3):
    D, I = index.search(query_embedding, k)  
    return I[0]  
