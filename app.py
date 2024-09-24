import streamlit as st
from encoder import encode_query
from generator import generate_answer
from indexer import search_passages
from utils import load_passages_from_file, load_embeddings
import json
import numpy
import faiss

passages = load_passages_from_file('data/processed_passages_new.txt')
print(f"Total number of loaded passages: {len(passages)}")

embeddings = load_embeddings('data/embeddings.pt')
index = faiss.read_index('data/faiss_index_HNSWFLAT.index')

st.title("Question Answering System")

query = st.text_input("Enter your question:")
if st.button("Submit"):
    if query:
        query_embedding = encode_query(query)
        closest_indices = search_passages(query_embedding, index)
        top_passages = [passages[i] for i in closest_indices]
        answer = generate_answer(query, top_passages)
        st.write("Generated Answer:", answer)
    else:
        st.write("Please enter a question.")
