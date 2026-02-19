"""
Part 3: Optimization Experiments
This script tests various optimization strategies mentioned in the assignment:
1. Different top-K values (1, 3, 5, 10)
2. Batch processing for vector search
3. Different index types (Flat vs IVF)
"""

# Set environment variables to prevent threading issues that cause segfaults
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from encode import QueryEncoder
from vector_db import VectorDatabase
from llm_generation import LLMGenerator
import faiss

def load_test_queries(queries_file="queries.json", num_queries=20):
    """Load test queries from the queries.json file."""
    with open(queries_file, 'r') as f:
        queries_data = json.load(f)
    
    # Extract query texts from list of dicts
    queries = []
    for query_item in queries_data:
        queries.append(query_item['text'])
        if len(queries) >= num_queries:
            break
    
    return queries

def experiment_topk_values(queries, top_k_values=[1, 3, 5, 10]):
    """Experiment with different top-K values."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: TOP-K VALUES")
    print("="*80)
    print(f"Testing top-K values: {top_k_values}")
    print("="*80 + "\n")
    
    results = {}
    
    for k in top_k_values:
        print(f"\n--- Testing top-K = {k} ---")
        
        # Initialize pipeline with this k value
        encoder = QueryEncoder()
        vector_db = VectorDatabase()
        vector_db.build_index("preprocessed_documents.json")
        llm = LLMGenerator("tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
        
        search_times = []
        retrieval_times = []
        llm_times = []
        total_times = []
        
        for i, query in enumerate(queries[:10], 1):  # Use first 10 queries
            print(f"  [{i}/10] Processing query...")
            
            # Encode
            query_embedding = encoder.encode(query)
            
            # Search
            start = time.time()
            distances, indices = vector_db.search(query_embedding, k=k)
            search_time = time.time() - start
            search_times.append(search_time)
            
            # Retrieve
            start = time.time()
            retrieved_docs = vector_db.get_documents_by_indices(indices)
            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)
            
            # Generate
            start = time.time()
            response = llm.generate_with_context(query, retrieved_docs, max_tokens=256)
            llm_time = time.time() - start
            llm_times.append(llm_time)
            
            total_times.append(search_time + retrieval_time + llm_time)
        
        results[k] = {
            'search_times': search_times,
            'retrieval_times': retrieval_times,
            'llm_times': llm_times,
            'total_times': total_times,
            'avg_search': np.mean(search_times),
            'avg_retrieval': np.mean(retrieval_times),
            'avg_llm': np.mean(llm_times),
            'avg_total': np.mean(total_times)
        }
        
        print(f"  Results for top-K={k}:")
        print(f"    Avg Search Time: {results[k]['avg_search']:.4f}s")
        print(f"    Avg Retrieval Time: {results[k]['avg_retrieval']:.4f}s")
        print(f"    Avg LLM Time: {results[k]['avg_llm']:.4f}s")
        print(f"    Avg Total Time: {results[k]['avg_total']:.4f}s")
    
    # Print comparison
    print("\n" + "="*80)
    print("TOP-K COMPARISON")
    print("="*80)
    print(f"{'Top-K':<10} {'Search':<12} {'Retrieval':<12} {'LLM Gen':<12} {'Total':<12}")
    print("-"*80)
    
    for k in top_k_values:
        print(f"{k:<10} {results[k]['avg_search']:>8.4f}s   {results[k]['avg_retrieval']:>8.4f}s   "
              f"{results[k]['avg_llm']:>8.4f}s   {results[k]['avg_total']:>8.4f}s")
    
    print("="*80)
    
    # Plot results
    plot_topk_comparison(results, top_k_values)
    
    return results

def experiment_batch_search(queries, batch_sizes=[1, 4, 8, 16, 32, 64]):
    """Experiment with batch processing for vector search."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: BATCH SEARCH")
    print("="*80)
    print(f"Testing batch sizes: {batch_sizes}")
    print("="*80 + "\n")
    
    encoder = QueryEncoder()
    vector_db = VectorDatabase()
    vector_db.build_index("preprocessed_documents.json")
    
    # Use first 128 queries for this experiment to properly test batch size 128
    test_queries = queries[:128]
    
    # Encode all queries first ONE AT A TIME to avoid memory issues
    print(f"  Encoding {len(test_queries)} queries...")
    query_embeddings = []
    for i, query in enumerate(test_queries):
        if (i + 1) % 10 == 0:
            print(f"    Encoded {i + 1}/{len(test_queries)} queries...")
        query_emb = encoder.encode(query)
        query_embeddings.append(query_emb)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    print(f"  ✓ Encoded {len(query_embeddings)} queries")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size = {batch_size} ---")
        
        try:
            # Process in batches
            total_search_time = 0
            num_batches = 0
            
            for i in range(0, len(test_queries), batch_size):
                batch_embeddings = query_embeddings[i:i+batch_size]
                
                start = time.time()
                # Search with batch
                if batch_embeddings.ndim == 1:
                    batch_embeddings = batch_embeddings.reshape(1, -1)
                distances, indices = vector_db.index.search(batch_embeddings, k=3)
                search_time = time.time() - start
                
                total_search_time += search_time
                num_batches += 1
        
            avg_time_per_query = total_search_time / len(test_queries)
            throughput = len(test_queries) / total_search_time
            
            results[batch_size] = {
                'total_time': total_search_time,
                'avg_time_per_query': avg_time_per_query,
                'throughput': throughput,
                'num_batches': num_batches
            }
            
            print(f"  Total search time: {total_search_time:.4f}s")
            print(f"  Avg time per query: {avg_time_per_query:.6f}s")
            print(f"  Throughput: {throughput:.2f} queries/sec")
            print(f"  Number of batches: {num_batches}")
            
        except Exception as e:
            print(f"  ✗ Error with batch size {batch_size}: {e}")
            print(f"  Skipping this batch size...")
            continue
    
    # Print comparison
    print("\n" + "="*80)
    print("BATCH SIZE COMPARISON")
    print("="*80)
    print(f"{'Batch Size':<12} {'Total Time':<15} {'Avg/Query':<15} {'Throughput':<15}")
    print("-"*80)
    
    for batch_size in batch_sizes:
        r = results[batch_size]
        print(f"{batch_size:<12} {r['total_time']:>10.4f}s    {r['avg_time_per_query']:>10.6f}s    "
              f"{r['throughput']:>10.2f} q/s")
    
    print("="*80)
    
    # Plot results
    plot_batch_comparison(results, batch_sizes)
    
    return results

def experiment_encoder_models(queries):
    """Experiment with different encoder models (different embedding dimensions)."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: ENCODER MODELS")
    print("="*80)
    print("Comparing different encoder models with different embedding dimensions")
    print("="*80 + "\n")
    
    import json
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    
    # Load original documents
    print("Loading original documents...")
    with open("documents.json", 'r') as f:
        documents = json.load(f)
    print(f"✓ Loaded {len(documents)} documents")
    
    # Test different encoder models
    encoder_configs = [
        {"name": "BGE-base", "model": "BAAI/bge-base-en-v1.5", "dim": 768},
        {"name": "MiniLM-L6", "model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384}
    ]
    
    results = {}
    
    for config in encoder_configs:
        print(f"\n--- Testing {config['name']} (dim={config['dim']}) ---")
        
        # Initialize encoder
        print(f"  Loading encoder model...")
        start = time.time()
        encoder_model = SentenceTransformer(config['model'])
        encoder_load_time = time.time() - start
        print(f"  ✓ Encoder loaded: {encoder_load_time:.4f}s")
        
        # Re-encode documents
        print(f"  Re-encoding {len(documents)} documents...")
        start = time.time()
        preprocessed_docs = []
        for doc in tqdm(documents, desc=f"  Encoding with {config['name']}"):
            embedding = encoder_model.encode(doc['text'], normalize_embeddings=True)
            preprocessed_docs.append({
                'id': doc['id'],
                'text': doc['text'],
                'embedding': embedding.tolist()
            })
        encoding_time = time.time() - start
        print(f"  ✓ Documents encoded: {encoding_time:.4f}s")
        
        # Build index
        print(f"  Building FAISS index...")
        embeddings_matrix = np.array([doc['embedding'] for doc in preprocessed_docs], dtype=np.float32)
        start = time.time()
        index = faiss.IndexFlatL2(config['dim'])
        index.add(embeddings_matrix)
        index_build_time = time.time() - start
        print(f"  ✓ Index built: {index_build_time:.4f}s")
        
        # Test queries
        print(f"  Testing with {len(queries[:10])} queries...")
        test_queries = queries[:10]
        
        encoding_times = []
        search_times = []
        total_times = []
        
        for i, query in enumerate(test_queries, 1):
            # Encode query
            start = time.time()
            query_emb = encoder_model.encode(query, normalize_embeddings=True)
            query_emb = np.array(query_emb, dtype=np.float32).reshape(1, -1)
            encoding_times.append(time.time() - start)
            
            # Search
            start = time.time()
            distances, indices = index.search(query_emb, k=3)
            search_times.append(time.time() - start)
            
            total_times.append(encoding_times[-1] + search_times[-1])
        
        results[config['name']] = {
            'dimension': config['dim'],
            'encoder_load_time': encoder_load_time,
            'document_encoding_time': encoding_time,
            'index_build_time': index_build_time,
            'avg_query_encoding': np.mean(encoding_times),
            'avg_search_time': np.mean(search_times),
            'avg_total_time': np.mean(total_times),
            'encoding_times': encoding_times,
            'search_times': search_times,
            'total_times': total_times
        }
        
        print(f"  Results:")
        print(f"    Avg Query Encoding: {results[config['name']]['avg_query_encoding']:.6f}s")
        print(f"    Avg Search Time: {results[config['name']]['avg_search_time']:.6f}s")
        print(f"    Avg Total Time: {results[config['name']]['avg_total_time']:.6f}s")
    
    # Print comparison
    print("\n" + "="*80)
    print("ENCODER MODEL COMPARISON")
    print("="*80)
    print(f"{'Encoder':<15} {'Dim':<8} {'Query Encode':<15} {'Search':<15} {'Total':<15}")
    print("-"*80)
    
    for name in [c['name'] for c in encoder_configs]:
        r = results[name]
        print(f"{name:<15} {r['dimension']:<8} {r['avg_query_encoding']:>10.6f}s    "
              f"{r['avg_search_time']:>10.6f}s    {r['avg_total_time']:>10.6f}s")
    
    print("="*80)
    
    # Plot results
    plot_encoder_comparison(results, encoder_configs)
    
    return results

def experiment_llm_generation(queries, max_tokens_values=[128, 256, 512, 1024]):
    """Experiment with different LLM generation parameters (max_tokens)."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: LLM GENERATION PARAMETERS")
    print("="*80)
    print(f"Testing different max_tokens values: {max_tokens_values}")
    print("="*80 + "\n")
    
    encoder = QueryEncoder()
    vector_db = VectorDatabase()
    vector_db.build_index("preprocessed_documents.json")
    
    results = {}
    
    for max_tokens in max_tokens_values:
        print(f"\n--- Testing max_tokens = {max_tokens} ---")
        
        # Initialize LLM for this configuration
        llm = LLMGenerator("tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
        
        llm_times = []
        response_lengths = []
        total_times = []
        
        test_queries = queries[:10]  # Use first 10 queries
        
        for i, query in enumerate(test_queries, 1):
            print(f"  [{i}/10] Processing query...")
            
            # Encode
            query_embedding = encoder.encode(query)
            
            # Search
            distances, indices = vector_db.search(query_embedding, k=3)
            retrieved_docs = vector_db.get_documents_by_indices(indices)
            
            # Generate with different max_tokens
            start = time.time()
            response = llm.generate_with_context(query, retrieved_docs, max_tokens=max_tokens)
            llm_time = time.time() - start
            
            llm_times.append(llm_time)
            response_lengths.append(len(response))
            
            # Total time (encoding + search + retrieval + LLM)
            total_times.append(llm_time)  # Just LLM time for this experiment
        
        results[max_tokens] = {
            'llm_times': llm_times,
            'response_lengths': response_lengths,
            'total_times': total_times,
            'avg_llm_time': np.mean(llm_times),
            'avg_response_length': np.mean(response_lengths),
            'std_llm_time': np.std(llm_times)
        }
        
        print(f"  Results for max_tokens={max_tokens}:")
        print(f"    Avg LLM Time: {results[max_tokens]['avg_llm_time']:.4f}s")
        print(f"    Avg Response Length: {results[max_tokens]['avg_response_length']:.0f} chars")
    
    # Print comparison
    print("\n" + "="*80)
    print("LLM GENERATION PARAMETER COMPARISON")
    print("="*80)
    print(f"{'Max Tokens':<12} {'Avg LLM Time':<15} {'Avg Response Len':<18} {'Std Dev':<12}")
    print("-"*80)
    
    for max_tokens in max_tokens_values:
        r = results[max_tokens]
        print(f"{max_tokens:<12} {r['avg_llm_time']:>10.4f}s    {r['avg_response_length']:>12.0f} chars      "
              f"{r['std_llm_time']:>8.4f}s")
    
    print("="*80)
    
    # Plot results
    plot_llm_comparison(results, max_tokens_values)
    
    return results

def experiment_ivf_index(queries):
    """Experiment with IVF index vs Flat index."""
    print("\n" + "="*80)
    print("EXPERIMENT 5: IVF INDEX vs FLAT INDEX")
    print("="*80)
    print("Comparing IndexFlatL2 vs IndexIVFFlat")
    print("="*80 + "\n")
    
    encoder = QueryEncoder()
    
    # Load documents
    with open("preprocessed_documents.json", 'r') as f:
        documents = json.load(f)
    
    embeddings = []
    for doc in documents:
        embeddings.append(doc['embedding'])
    
    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    dimension = 768
    
    # Test queries - encode one at a time to avoid segfault
    test_queries = queries[:20]
    print(f"  Encoding {len(test_queries)} queries...")
    query_embeddings = []
    for query in test_queries:
        query_emb = encoder.encode(query)
        query_embeddings.append(query_emb)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    
    results = {}
    
    # Test 1: Flat Index
    print("\n--- Testing Flat Index ---")
    start = time.time()
    flat_index = faiss.IndexFlatL2(dimension)
    flat_index.add(embeddings_matrix)
    flat_build_time = time.time() - start
    print(f"  Build time: {flat_build_time:.4f}s")
    
    flat_search_times = []
    for query_emb in query_embeddings:
        query_emb = query_emb.reshape(1, -1)
        start = time.time()
        distances, indices = flat_index.search(query_emb, k=3)
        flat_search_times.append(time.time() - start)
    
    results['flat'] = {
        'build_time': flat_build_time,
        'avg_search_time': np.mean(flat_search_times),
        'total_search_time': sum(flat_search_times),
        'search_times': flat_search_times
    }
    
    print(f"  Avg search time: {results['flat']['avg_search_time']:.6f}s")
    print(f"  Total search time: {results['flat']['total_search_time']:.4f}s")
    
    # Test 2: IVF Index
    print("\n--- Testing IVF Index ---")
    nlist = 100  # Number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    
    start = time.time()
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Train the index
    print(f"  Training IVF index with {nlist} clusters...")
    ivf_index.train(embeddings_matrix)
    ivf_index.add(embeddings_matrix)
    ivf_build_time = time.time() - start
    print(f"  Build time: {ivf_build_time:.4f}s")
    
    # Set search parameters
    ivf_index.nprobe = 10  # Number of clusters to search
    
    ivf_search_times = []
    for query_emb in query_embeddings:
        query_emb = query_emb.reshape(1, -1)
        start = time.time()
        distances, indices = ivf_index.search(query_emb, k=3)
        ivf_search_times.append(time.time() - start)
    
    results['ivf'] = {
        'build_time': ivf_build_time,
        'avg_search_time': np.mean(ivf_search_times),
        'total_search_time': sum(ivf_search_times),
        'search_times': ivf_search_times,
        'nlist': nlist,
        'nprobe': ivf_index.nprobe
    }
    
    print(f"  Avg search time: {results['ivf']['avg_search_time']:.6f}s")
    print(f"  Total search time: {results['ivf']['total_search_time']:.4f}s")
    
    # Print comparison
    print("\n" + "="*80)
    print("INDEX TYPE COMPARISON")
    print("="*80)
    print(f"{'Index Type':<15} {'Build Time':<15} {'Avg Search':<15} {'Total Search':<15}")
    print("-"*80)
    print(f"{'Flat':<15} {results['flat']['build_time']:>10.4f}s    "
          f"{results['flat']['avg_search_time']:>10.6f}s    {results['flat']['total_search_time']:>10.4f}s")
    print(f"{'IVF':<15} {results['ivf']['build_time']:>10.4f}s    "
          f"{results['ivf']['avg_search_time']:>10.6f}s    {results['ivf']['total_search_time']:>10.4f}s")
    
    speedup = results['flat']['avg_search_time'] / results['ivf']['avg_search_time']
    print(f"\nIVF speedup: {speedup:.2f}x")
    print("="*80)
    
    # Plot results
    plot_index_comparison(results)
    
    return results

def plot_topk_comparison(results, top_k_values, output_file="topk_comparison.png"):
    """Plot comparison of different top-K values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Top-K Value Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Component times
    search_times = [results[k]['avg_search'] for k in top_k_values]
    retrieval_times = [results[k]['avg_retrieval'] for k in top_k_values]
    llm_times = [results[k]['avg_llm'] for k in top_k_values]
    
    x = np.arange(len(top_k_values))
    width = 0.25
    
    ax1.bar(x - width, search_times, width, label='Search', color='#FF6B6B')
    ax1.bar(x, retrieval_times, width, label='Retrieval', color='#4ECDC4')
    ax1.bar(x + width, llm_times, width, label='LLM Gen', color='#45B7D1')
    
    ax1.set_xlabel('Top-K Value')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Component Times by Top-K')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_k_values)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Total time
    total_times = [results[k]['avg_total'] for k in top_k_values]
    ax2.plot(top_k_values, total_times, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax2.set_xlabel('Top-K Value')
    ax2.set_ylabel('Total Time (seconds)')
    ax2.set_title('Total Query Time by Top-K')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")

def plot_batch_comparison(results, batch_sizes, output_file="batch_comparison.png"):
    """Plot comparison of different batch sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Batch Size Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Throughput
    throughputs = [results[bs]['throughput'] for bs in batch_sizes]
    ax1.plot(batch_sizes, throughputs, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (queries/second)')
    ax1.set_title('Throughput vs Batch Size')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average time per query
    avg_times = [results[bs]['avg_time_per_query'] * 1000 for bs in batch_sizes]  # Convert to ms
    ax2.plot(batch_sizes, avg_times, marker='o', linewidth=2, markersize=8, color='#45B7D1')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Avg Time per Query (ms)')
    ax2.set_title('Latency vs Batch Size')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")

def plot_encoder_comparison(results, encoder_configs, output_file="encoder_comparison.png"):
    """Plot comparison of different encoder models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Encoder Model Comparison', fontsize=16, fontweight='bold')
    
    encoder_names = [c['name'] for c in encoder_configs]
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Plot 1: Query encoding time
    encoding_times = [results[name]['avg_query_encoding'] * 1000 for name in encoder_names]  # Convert to ms
    
    bars1 = ax1.bar(encoder_names, encoding_times, color=colors, edgecolor='black')
    ax1.set_ylabel('Time (milliseconds)')
    ax1.set_title('Average Query Encoding Time')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}ms', ha='center', va='bottom')
    
    # Plot 2: Total time (encoding + search)
    total_times = [results[name]['avg_total_time'] * 1000 for name in encoder_names]  # Convert to ms
    
    bars2 = ax2.bar(encoder_names, total_times, color=colors, edgecolor='black')
    ax2.set_ylabel('Time (milliseconds)')
    ax2.set_title('Average Total Time (Encoding + Search)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")

def plot_llm_comparison(results, max_tokens_values, output_file="llm_comparison.png"):
    """Plot comparison of different LLM generation parameters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LLM Generation Parameter Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Generation time vs max_tokens
    avg_times = [results[mt]['avg_llm_time'] for mt in max_tokens_values]
    ax1.plot(max_tokens_values, avg_times, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax1.set_xlabel('Max Tokens')
    ax1.set_ylabel('Average Generation Time (seconds)')
    ax1.set_title('Generation Time vs Max Tokens')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Response length vs max_tokens
    avg_lengths = [results[mt]['avg_response_length'] for mt in max_tokens_values]
    ax2.plot(max_tokens_values, avg_lengths, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
    ax2.set_xlabel('Max Tokens')
    ax2.set_ylabel('Average Response Length (characters)')
    ax2.set_title('Response Length vs Max Tokens')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")

def plot_index_comparison(results, output_file="index_comparison.png"):
    """Plot comparison of Flat vs IVF index."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Index Type Comparison: Flat vs IVF', fontsize=16, fontweight='bold')
    
    # Plot 1: Build time
    index_types = ['Flat', 'IVF']
    build_times = [results['flat']['build_time'], results['ivf']['build_time']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars1 = ax1.bar(index_types, build_times, color=colors, edgecolor='black')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Index Build Time')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')
    
    # Plot 2: Search time
    search_times = [results['flat']['avg_search_time'] * 1000, 
                    results['ivf']['avg_search_time'] * 1000]  # Convert to ms
    
    bars2 = ax2.bar(index_types, search_times, color=colors, edgecolor='black')
    ax2.set_ylabel('Time (milliseconds)')
    ax2.set_title('Average Search Time per Query')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")

def main():
    """Main function for optimization experiments."""
    print("\n" + "="*80)
    print("CS4414 HW3 - PART 3: OPTIMIZATION EXPERIMENTS")
    print("="*80)
    
    # Load test queries
    print("\nLoading test queries...")
    queries = load_test_queries("queries.json", num_queries=64)
    print(f"✓ Loaded {len(queries)} queries")
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING OPTIMIZATION EXPERIMENTS")
    print("="*80)
    
    # Experiment 1: Different top-K values
    topk_results = experiment_topk_values(queries, top_k_values=[1, 3, 5, 10])
    
    # Experiment 2: Batch search
    # Using smaller batch sizes to avoid memory issues
    batch_results = experiment_batch_search(queries, batch_sizes=[1, 4, 8, 16, 32, 64, 128])
    
    # Experiment 3: Different encoder models
    encoder_results = experiment_encoder_models(queries)
    
    # Experiment 4: LLM generation parameters
    llm_results = experiment_llm_generation(queries, max_tokens_values=[128, 256, 512, 1024])
    
    # Experiment 5: IVF vs Flat index
    index_results = experiment_ivf_index(queries)
    
    # Save all results
    print("\n" + "="*80)
    print("SAVING EXPERIMENT RESULTS")
    print("="*80)
    
    all_results = {
        'topk_experiment': topk_results,
        'batch_experiment': batch_results,
        'encoder_experiment': encoder_results,
        'llm_experiment': llm_results,
        'index_experiment': index_results
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    all_results = convert_to_serializable(all_results)
    
    with open("optimization_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("✓ Saved results to optimization_results.json")
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - topk_comparison.png")
    print("  - batch_comparison.png")
    print("  - encoder_comparison.png")
    print("  - llm_comparison.png")
    print("  - index_comparison.png")
    print("  - optimization_results.json")
    print("\nUse these results for your Part 3 optimization analysis!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

