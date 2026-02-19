"""
Test script for the RAG pipeline.
Tests each component individually and then the full pipeline.
"""
import os
import sys
import json


def test_encoder():
    """Test Component 1: Query Encoder"""
    print("\n" + "="*60)
    print("Testing Component 1: Query Encoder")
    print("="*60)
    
    from encode import QueryEncoder
    
    encoder = QueryEncoder()
    
    # Test single query
    test_query = "What causes squirrels to lose fur?"
    embedding = encoder.encode(test_query)
    
    assert embedding.shape == (768,), f"Expected shape (768,), got {embedding.shape}"
    assert str(embedding.dtype) == 'float32', f"Expected dtype float32, got {embedding.dtype}"
    
    print(f"✓ Single query encoding works")
    print(f"  Query: {test_query}")
    print(f"  Embedding shape: {embedding.shape}")
    
    # Test batch encoding
    test_queries = ["Query 1", "Query 2", "Query 3"]
    batch_embeddings = encoder.encode_batch(test_queries)
    
    assert batch_embeddings.shape == (3, 768), f"Expected shape (3, 768), got {batch_embeddings.shape}"
    
    print(f"✓ Batch encoding works")
    print(f"  Number of queries: {len(test_queries)}")
    print(f"  Batch embeddings shape: {batch_embeddings.shape}")
    
    return encoder


def test_vector_database():
    """Test Components 2-3: Vector Database and Document Retrieval"""
    print("\n" + "="*60)
    print("Testing Components 2-3: Vector Database & Document Retrieval")
    print("="*60)
    
    from vector_db import VectorDatabase
    
    vdb = VectorDatabase()
    vdb.build_index("preprocessed_documents.json")
    
    print(f"✓ Vector database built")
    print(f"  Number of documents: {len(vdb.documents)}")
    
    # Test self-similarity search
    test_idx = 42
    test_embedding = vdb.documents[test_idx]['embedding']
    distances, indices = vdb.search(test_embedding, k=5)
    
    assert distances.shape == (1, 5), f"Expected distances shape (1, 5), got {distances.shape}"
    assert indices.shape == (1, 5), f"Expected indices shape (1, 5), got {indices.shape}"
    assert indices[0][0] == test_idx, f"Top result should be document {test_idx}, got {indices[0][0]}"
    assert distances[0][0] < 0.001, f"Distance to self should be ~0, got {distances[0][0]}"
    
    print(f"✓ Vector search works (self-similarity test passed)")
    print(f"  Test document index: {test_idx}")
    print(f"  Top result index: {indices[0][0]}")
    print(f"  Distance to self: {distances[0][0]:.6f}")
    
    # Test document retrieval
    retrieved_docs = vdb.get_documents_by_indices(indices)
    
    assert len(retrieved_docs) == 5, f"Expected 5 documents, got {len(retrieved_docs)}"
    assert 'id' in retrieved_docs[0], "Document should have 'id' field"
    assert 'text' in retrieved_docs[0], "Document should have 'text' field"
    
    print(f"✓ Document retrieval works")
    print(f"  Retrieved {len(retrieved_docs)} documents")
    print(f"  First document ID: {retrieved_docs[0]['id']}")
    print(f"  First document text preview: {retrieved_docs[0]['text'][:100]}...")
    
    return vdb


def test_llm_generator():
    """Test Component 5: LLM Generation"""
    print("\n" + "="*60)
    print("Testing Component 5: LLM Generation")
    print("="*60)
    
    llm_model = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    if not os.path.exists(llm_model):
        print(f"⚠️  Skipping LLM test - model not found: {llm_model}")
        print(f"   Download it with:")
        print(f"   wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
        return None
    
    from llm_generation import LLMGenerator
    
    generator = LLMGenerator(llm_model)
    
    # Test simple generation
    test_prompt = "What is the capital of France?"
    response = generator.generate(test_prompt, max_tokens=50)
    
    assert isinstance(response, str), f"Expected string response, got {type(response)}"
    assert len(response) > 0, "Response should not be empty"
    
    print(f"✓ Basic LLM generation works")
    print(f"  Prompt: {test_prompt}")
    print(f"  Response: {response[:100]}...")
    
    # Test with context
    test_query = "What causes squirrels to lose fur?"
    test_docs = [
        {'id': 0, 'text': "Squirrels may lose fur due to mange, caused by mites."},
        {'id': 1, 'text': "Fungal infections can also cause fur loss in squirrels."}
    ]
    
    response = generator.generate_with_context(test_query, test_docs, max_tokens=100)
    
    assert isinstance(response, str), f"Expected string response, got {type(response)}"
    assert len(response) > 0, "Response should not be empty"
    
    print(f"✓ LLM generation with context works")
    print(f"  Query: {test_query}")
    print(f"  Response: {response[:100]}...")
    
    return generator


def test_full_pipeline():
    """Test Component 6: Full RAG Pipeline"""
    print("\n" + "="*60)
    print("Testing Component 6: Full RAG Pipeline")
    print("="*60)
    
    llm_model = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    if not os.path.exists(llm_model):
        print(f"⚠️  Skipping full pipeline test - LLM model not found")
        return
    
    from main import RAGPipeline
    
    pipeline = RAGPipeline(
        preprocessed_docs="preprocessed_documents.json",
        llm_model_path=llm_model,
        top_k=3
    )
    
    # Test with sample queries
    test_queries = [
        "What causes squirrels to lose fur?",
        "How to improve computer performance?"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: {query} ---")
        result = pipeline.query(query, verbose=False)
        
        assert 'query' in result, "Result should contain 'query'"
        assert 'retrieved_docs' in result, "Result should contain 'retrieved_docs'"
        assert 'response' in result, "Result should contain 'response'"
        assert len(result['retrieved_docs']) == 3, f"Expected 3 docs, got {len(result['retrieved_docs'])}"
        
        print(f"✓ Query processed successfully")
        print(f"  Retrieved {len(result['retrieved_docs'])} documents")
        print(f"  Response length: {len(result['response'])} characters")
        print(f"  Response preview: {result['response'][:150]}...")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CS4414 HW3 Part 2 - RAG Pipeline Test Suite")
    print("="*60)
    
    # Check prerequisites
    if not os.path.exists("preprocessed_documents.json"):
        print("\n❌ ERROR: preprocessed_documents.json not found!")
        print("Please run data_preprocess.py first.")
        sys.exit(1)
    
    try:
        # Test each component
        encoder = test_encoder()
        vdb = test_vector_database()
        llm = test_llm_generator()
        
        if llm is not None:
            test_full_pipeline()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\nYour RAG pipeline is ready to use!")
        print("Run: python main.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

