from PIL import Image
from sentence_transformers import SentenceTransformer, util

class MultimodalSearch:
    # Move documents to the first position
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        
        self.texts = [
            f"{doc['title']}: {doc['description']}" 
            for doc in self.documents
        ]
        
        print(f"Encoding {len(self.texts)} movie descriptions...")
        self.text_embeddings = self.model.encode(
            self.texts, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )

    def embed_image(self, image_path: str):
        """
        Loads an image from a path and returns its vector embedding.
        """
        image = Image.open(image_path)
        
        embedding = self.model.encode([image])[0]
        return embedding

    def search_with_image(self, image_path: str):
        image_embedding = self.model.encode(
            [Image.open(image_path)], 
            convert_to_tensor=True
        )[0]
        
        similarities = util.cos_sim(image_embedding, self.text_embeddings)[0]
        
        results = []
        for i, score in enumerate(similarities):
            results.append({
                "id": self.documents[i].get("id"),
                "title": self.documents[i].get("title"),
                "description": self.documents[i].get("description"),
                "score": score.item()
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:5]

def verify_image_embedding(image_path: str):
    """
    High-level function to verify that the embedding pipeline is working.
    """
    search_engine = MultimodalSearch()
    embedding = search_engine.embed_image(image_path)
    
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    # This assumes you have a helper to load your movie dataset
    # For this example, we'll assume 'load_movies()' exists or you provide a list
    from lib.search_utils import load_movies # Placeholder for your data loading logic
    movies = load_movies() 
    
    search_engine = MultimodalSearch(movies)
    return search_engine.search_with_image(image_path)