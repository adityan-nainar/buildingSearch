from sentence_transformers import SentenceTransformer
import numpy as np
import os
from lib.search_utils import load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Empty text")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar= True)
        np.save('cache/movie_embeddings.npy', self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings= np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents): return self.embeddings
        else: 
            self.embeddings = self.build_embeddings(documents)
            return self.embeddings
    
    def search(self, query, limit):
        if len(self.embeddings)==0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        tuple_list = []
        for i in range(len(self.embeddings)):
            similarity_score = cosine_similarity(query_embedding, self.embeddings[i])
            tuple_list.append((similarity_score, self.document_map[self.documents[i]["id"]]))
        sortedlist = sorted(tuple_list, key=lambda x: x[0], reverse=True)
        return sortedlist[:limit]

                    
def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}") 
    print(f"Max sequence length: {search_instance.model.max_seq_length}")
        
def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")   


def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)