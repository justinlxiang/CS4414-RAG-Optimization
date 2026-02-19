import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


def load_documents(input_file):
    """Load documents from JSON file."""
    with open(input_file, 'r') as f:
        documents = json.load(f)
    return documents


def initialize_encoder(model_name):
    """Initialize the BGE encoder model."""
    bge_model = SentenceTransformer(model_name)
    return bge_model


def preprocess_documents(documents, bge_model, output_file):
    """
    Process all documents: encode each and save results to JSON specified in output_file.
    """
    preprocessed_docs = []
    for doc in tqdm(documents, desc="Encoding documents"):
        doc_id = doc['id']
        text = doc['text']
        
        embedding = bge_model.encode(text, normalize_embeddings=True)
        embedding_list = embedding.tolist()
        preprocessed_doc = {
            'id': doc_id,
            'text': text,
            'embedding': embedding_list
        }
        
        preprocessed_docs.append(preprocessed_doc)
    
    with open(output_file, 'w') as f:
        json.dump(preprocessed_docs, f, indent=2)
        

def main():
    input_file = "documents.json"
    model_name = "BAAI/bge-base-en-v1.5"
    output_file = "preprocessed_documents.json"
    
    documents = load_documents(input_file)
    bge_model = initialize_encoder(model_name)
    preprocess_documents(documents, bge_model, output_file)

if __name__ == "__main__":
    main()

