import json
import numpy as np
import faiss

class VectorDatabase:
    """Vector database for semantic document search using FAISS."""
    def __init__(self):
        self.index = None
        self.documents = []
        self.dimension = 768
    
    def build_index(self, preprocessed_file):
        """Build FAISS index from document embeddings."""
        with open(preprocessed_file, 'r') as f:
            self.documents = json.load(f)

        embeddings = []
        for doc in self.documents:
            embeddings.append(doc['embedding'])
        
        
        embeddings_matrix = np.array(embeddings, dtype=np.float32) # Shape is (n_documents, dimension)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_matrix)
    
    def search(self, query_embedding, k=3):
        """
        Search for top-k most diverse documents (unique by document ID).
        
        Multiple embeddings may have the same document ID (different chunks of same doc).
        This method deduplicates results to return k documents with unique IDs.
 
        Returns:
            distances: Array of shape (1, k) with L2 distances
            indices: Array of shape (1, k) with document indices
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Convert query to numpy array and ensure correct shape (batch_size, dimension)
        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)  # Shape: (1, 768)
        
        # Search for more results than needed to account for duplicates
        search_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)
        
        seen_ids = set()
        unique_distances = []
        unique_indices = []
        
        for i in range(search_k):
            idx = indices[0][i]
            document_id = self.documents[idx]['id']
            
            if document_id not in seen_ids:
                seen_ids.add(document_id)
                unique_distances.append(distances[0][i])
                unique_indices.append(idx)
                
                if len(unique_indices) == k:
                    break
        
        # If we still didn't find enough unique documents, search again with k * 10
        if len(unique_indices) < k:
            search_k = min(k * 10, self.index.ntotal)
            distances, indices = self.index.search(query, search_k)
            
            for i in range(search_k):
                idx = indices[0][i]
                document_id = self.documents[idx]['id']
                
                if document_id not in seen_ids:
                    seen_ids.add(document_id)
                    unique_distances.append(distances[0][i])
                    unique_indices.append(idx)
                    
                    if len(unique_indices) == k:
                        break
        
        final_distances = np.array([unique_distances], dtype=np.float32) # Shape is (1, k)
        final_indices = np.array([unique_indices], dtype=np.int64) # Shape is (1, k)
        
        return final_distances, final_indices
    
    def get_documents_by_indices(self, indices):
        # Handle both 1D and 2D arrays
        if isinstance(indices, np.ndarray):
            if indices.ndim == 2:
                indices = indices[0]  # Extract first row if 2D
            indices = indices.tolist()
        
        retrieved_docs = []
        for idx in indices:
            doc = self.documents[idx]
            retrieved_docs.append({
                'id': doc['id'],
                'text': doc['text']
            })
        
        return retrieved_docs

def main():
    vdb = VectorDatabase()
    vdb.build_index("preprocessed_documents.json")
    
    print("\n=== Step 3: Testing Vector Database ===")
    test_doc_idx = 42
    test_doc = vdb.documents[test_doc_idx]
    
    print(f"\nTest Query: Using document at index {test_doc_idx}")
    print(f"Document ID: {test_doc['id']}")
    print(f"Document text: {test_doc['text'][:200]}...")
    
    distances, indices = vdb.search(test_doc['embedding'], k=5)
    
    print(f"\nTop 5 search results:")
    print(f"Distances: {distances[0]}")
    print(f"Indices: {indices[0]}")
    
    print(f"\nTop 5 results:")
    doc_ids_seen = []
    for i in range(5):
        result_idx = indices[0][i]
        result_doc = vdb.documents[result_idx]
        doc_ids_seen.append(result_doc['id'])
        print(f"{i+1}. Index: {result_idx}, ID: {result_doc['id']}, Distance: {distances[0][i]:.4f}")
        print(f"   Text: {result_doc['text'][:200]}...")
    
    if len(doc_ids_seen) == len(set(doc_ids_seen)):
        print(f"\nAll {len(doc_ids_seen)} results have unique document IDs (deduplication working)")
    else:
        print(f"\nWARNING: Duplicate document IDs found!")


if __name__ == "__main__":
    main()

