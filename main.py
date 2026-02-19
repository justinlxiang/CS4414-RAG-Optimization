import time
import numpy as np
from encode import QueryEncoder
from vector_db import VectorDatabase
from llm_generation import LLMGenerator

class RAGPipeline:    
    def __init__(self, 
                 preprocessed_docs="preprocessed_documents.json",
                 llm_model_path="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
                 top_k=3):
        
        self.top_k = top_k
        
        print("\n" + "="*80)
        print("INITIALIZING RAG PIPELINE")
        print("="*80)
        
        # Time encoder initialization
        start = time.time()
        self.encoder = QueryEncoder()
        encoder_time = time.time() - start
        print(f"✓ Encoder loaded: {encoder_time:.4f}s")
        
        # Time vector database initialization
        start = time.time()
        self.vector_db = VectorDatabase()
        self.vector_db.build_index(preprocessed_docs)
        vdb_time = time.time() - start
        print(f"✓ Vector database built: {vdb_time:.4f}s ({len(self.vector_db.documents)} documents)")
        
        # Time LLM initialization
        start = time.time()
        self.llm = LLMGenerator(llm_model_path)
        llm_time = time.time() - start
        print(f"✓ LLM loaded: {llm_time:.4f}s")
        
        total_init_time = encoder_time + vdb_time + llm_time
        print(f"\nTotal initialization time: {total_init_time:.4f}s")
        print("="*80 + "\n")
        
        # Track timing statistics across queries
        self.timing_stats = {
            'query_encoding': [],
            'vector_search': [],
            'document_retrieval': [],
            'prompt_augmentation': [],
            'llm_generation': [],
            'total_query_time': []
        }
    
    def query(self, user_question, verbose=True):
        """Execute RAG pipeline with detailed timing."""
        
        if verbose:
            print("\n" + "="*80)
            print("QUERY PROCESSING PIPELINE")
            print("="*80)
            print(f"Query: {user_question}\n")
        
        total_start = time.time()
        timings = {}
        
        # Step 1: Query Encoding
        if verbose:
            print("Step 1: Query Encoding...")
        start = time.time()
        query_embedding = self.encoder.encode(user_question)
        timings['query_encoding'] = time.time() - start
        if verbose:
            print(f"  ✓ Encoded query to {len(query_embedding)}-dim vector: {timings['query_encoding']:.4f}s\n")
        
        # Step 2: Vector Search
        if verbose:
            print("Step 2: Vector Search...")
        start = time.time()
        distances, indices = self.vector_db.search(query_embedding, k=self.top_k)
        timings['vector_search'] = time.time() - start
        if verbose:
            print(f"  ✓ Found top-{self.top_k} similar documents: {timings['vector_search']:.4f}s")
            print(f"  Distances: {[f'{d:.2f}' for d in distances[0]]}\n")
        
        # Step 3: Document Retrieval
        if verbose:
            print("Step 3: Document Retrieval...")
        start = time.time()
        retrieved_docs = self.vector_db.get_documents_by_indices(indices)
        timings['document_retrieval'] = time.time() - start
        if verbose:
            print(f"  ✓ Retrieved {len(retrieved_docs)} document texts: {timings['document_retrieval']:.4f}s")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"    Doc {i} (ID {doc['id']}): {doc['text'][:80]}...")
            print()
        
        # Step 4: Prompt Augmentation
        if verbose:
            print("Step 4: Prompt Augmentation...")
        start = time.time()
        augmented_prompt = self.llm.create_augmented_prompt(user_question, retrieved_docs)
        timings['prompt_augmentation'] = time.time() - start
        if verbose:
            print(f"  ✓ Created augmented prompt ({len(augmented_prompt)} chars): {timings['prompt_augmentation']:.4f}s\n")
        
        # Step 5: LLM Generation
        if verbose:
            print("Step 5: LLM Generation...")
        start = time.time()
        response = self.llm.generate(augmented_prompt, max_tokens=256)
        timings['llm_generation'] = time.time() - start
        if verbose:
            print(f"  ✓ Generated response ({len(response)} chars): {timings['llm_generation']:.4f}s\n")
        
        timings['total_query_time'] = time.time() - total_start
        
        # Update statistics
        for key, value in timings.items():
            self.timing_stats[key].append(value)
        
        if verbose:
            self.print_timing_summary(timings)
        
        return {
            'query': user_question,
            'retrieved_docs': retrieved_docs,
            'distances': distances[0].tolist(),
            'response': response,
            'timings': timings
        }
    
    def print_timing_summary(self, timings):
        """Print detailed timing breakdown."""
        print("="*80)
        print("TIMING BREAKDOWN")
        print("="*80)
        
        total = timings['total_query_time']
        
        components = [
            ('Query Encoding', timings['query_encoding']),
            ('Vector Search', timings['vector_search']),
            ('Document Retrieval', timings['document_retrieval']),
            ('Prompt Augmentation', timings['prompt_augmentation']),
            ('LLM Generation', timings['llm_generation'])
        ]
        
        print(f"{'Component':<25} {'Time (s)':<12} {'% of Total':<12} {'Bar'}")
        print("-"*80)
        
        for name, time_val in components:
            percentage = (time_val / total) * 100
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            print(f"{name:<25} {time_val:>8.4f}s    {percentage:>6.2f}%      {bar}")
        
        print("-"*80)
        print(f"{'TOTAL':<25} {total:>8.4f}s    {100.0:>6.2f}%")
        print("="*80 + "\n")
    
    def print_aggregate_statistics(self):
        """Print aggregate statistics across all queries."""
        if not self.timing_stats['total_query_time']:
            print("No queries executed yet.")
            return
        
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS ACROSS ALL QUERIES")
        print("="*80)
        print(f"Total queries executed: {len(self.timing_stats['total_query_time'])}\n")
        
        print(f"{'Component':<25} {'Mean (s)':<12} {'Std (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        print("-"*80)
        
        for component, times in self.timing_stats.items():
            if times:
                mean = np.mean(times)
                std = np.std(times)
                min_val = np.min(times)
                max_val = np.max(times)
                print(f"{component:<25} {mean:>8.4f}     {std:>8.4f}     {min_val:>8.4f}     {max_val:>8.4f}")
        
        print("="*80)
        
        # Print average percentage breakdown
        print("\nAVERAGE PERCENTAGE BREAKDOWN:")
        print("-"*80)
        avg_total = np.mean(self.timing_stats['total_query_time'])
        
        components = [
            ('Query Encoding', self.timing_stats['query_encoding']),
            ('Vector Search', self.timing_stats['vector_search']),
            ('Document Retrieval', self.timing_stats['document_retrieval']),
            ('Prompt Augmentation', self.timing_stats['prompt_augmentation']),
            ('LLM Generation', self.timing_stats['llm_generation'])
        ]
        
        for name, times in components:
            avg_time = np.mean(times)
            percentage = (avg_time / avg_total) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            print(f"{name:<25} {percentage:>6.2f}%      {bar}")
        
        print("="*80 + "\n")
                
def main():
    preprocessed_docs = "preprocessed_documents.json"
    llm_model = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    
    # Initialize the RAG pipeline
    pipeline = RAGPipeline(
        preprocessed_docs=preprocessed_docs,
        llm_model_path=llm_model,
        top_k=3
    )
    
    print("\n" + "="*80)
    print("RAG SYSTEM READY - Interactive Mode")
    print("="*80)
    print("Commands:")
    print("  - Type your question to get an AI response")
    print("  - Type 'stats' to see aggregate statistics")
    print("  - Type 'quit' or 'exit' to exit")
    print("="*80)
    
    while True:
        user_input = input("\n>>> What is your question? ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            pipeline.print_aggregate_statistics()
            break
        
        if user_input.lower() == 'stats':
            pipeline.print_aggregate_statistics()
            continue
        
        if not user_input:
            continue
        
        result = pipeline.query(user_input, verbose=True)
        
        print("\n" + "="*80)
        print("AI RESPONSE")
        print("="*80)
        print(result['response'])
        print("="*80)


if __name__ == "__main__":
    main()

