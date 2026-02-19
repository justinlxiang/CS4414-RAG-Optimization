import numpy as np
from sentence_transformers import SentenceTransformer

class QueryEncoder:
    """Encodes text queries into embedding vectors."""
    
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 768
    
    def encode(self, query_text):
        embedding = self.model.encode(query_text, normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)
        return embedding
    
    def encode_batch(self, query_texts):
        embeddings = self.model.encode(query_texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings


def main():
    """Test the query encoder."""
    encoder = QueryEncoder()
    
    # Test with a sample query
    test_query = "What causes squirrels to lose fur?"
    print(f"\nTest Query: {test_query}")
    
    embedding = encoder.encode(test_query)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Embedding dtype: {embedding.dtype}")
    
    # Test batch encoding
    test_queries = [
        "What causes squirrels to lose fur?",
        "How to improve computer performance?",
        "What is the weather like today?"
    ]
    print(f"\n\nBatch encoding {len(test_queries)} queries...")
    batch_embeddings = encoder.encode_batch(test_queries)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")


if __name__ == "__main__":
    main()

