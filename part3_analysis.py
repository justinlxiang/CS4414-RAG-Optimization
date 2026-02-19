"""
Part 3: System Analysis and Performance Benchmarking
This script runs multiple queries through the RAG pipeline and collects detailed timing statistics.
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from main import RAGPipeline

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

def run_benchmark(pipeline, queries, warmup=2):
    """Run benchmark on multiple queries."""
    print("\n" + "="*80)
    print("RUNNING BENCHMARK")
    print("="*80)
    print(f"Total queries: {len(queries)}")
    print(f"Warmup queries: {warmup}")
    print("="*80 + "\n")
    
    results = []
    
    # Warmup runs (not counted in statistics)
    if warmup > 0:
        print(f"Running {warmup} warmup queries...")
        for i in range(min(warmup, len(queries))):
            pipeline.query(queries[i], verbose=False)
        print("Warmup complete.\n")
    
    # Actual benchmark runs
    print(f"Running {len(queries)} benchmark queries...\n")
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Processing: {query[:60]}...")
        result = pipeline.query(query, verbose=False)
        results.append(result)
        
        # Print brief timing info
        timings = result['timings']
        print(f"  Total: {timings['total_query_time']:.4f}s | "
              f"Encode: {timings['query_encoding']:.4f}s | "
              f"Search: {timings['vector_search']:.4f}s | "
              f"LLM: {timings['llm_generation']:.4f}s\n")
    
    return results

def analyze_results(results):
    """Analyze and print detailed statistics."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Extract timing data
    timing_data = {
        'Query Encoding': [],
        'Vector Search': [],
        'Document Retrieval': [],
        'Prompt Augmentation': [],
        'LLM Generation': [],
        'Total': []
    }
    
    for result in results:
        t = result['timings']
        timing_data['Query Encoding'].append(t['query_encoding'])
        timing_data['Vector Search'].append(t['vector_search'])
        timing_data['Document Retrieval'].append(t['document_retrieval'])
        timing_data['Prompt Augmentation'].append(t['prompt_augmentation'])
        timing_data['LLM Generation'].append(t['llm_generation'])
        timing_data['Total'].append(t['total_query_time'])
    
    # Print statistics table
    print(f"\n{'Component':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-"*80)
    
    for component, times in timing_data.items():
        mean = np.mean(times)
        std = np.std(times)
        min_val = np.min(times)
        max_val = np.max(times)
        median = np.median(times)
        print(f"{component:<25} {mean:>8.4f}s  {std:>8.4f}s  {min_val:>8.4f}s  {max_val:>8.4f}s  {median:>8.4f}s")
    
    # Print percentage breakdown
    print("\n" + "="*80)
    print("AVERAGE TIME BREAKDOWN (Percentage of Total)")
    print("="*80)
    
    avg_total = np.mean(timing_data['Total'])
    
    for component in ['Query Encoding', 'Vector Search', 'Document Retrieval', 
                      'Prompt Augmentation', 'LLM Generation']:
        avg_time = np.mean(timing_data[component])
        percentage = (avg_time / avg_total) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{component:<25} {avg_time:>8.4f}s  {percentage:>6.2f}%  {bar}")
    
    print("-"*80)
    print(f"{'TOTAL':<25} {avg_total:>8.4f}s  {100.0:>6.2f}%")
    print("="*80)
    
    # Identify bottleneck
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    component_times = {
        'Query Encoding': np.mean(timing_data['Query Encoding']),
        'Vector Search': np.mean(timing_data['Vector Search']),
        'Document Retrieval': np.mean(timing_data['Document Retrieval']),
        'Prompt Augmentation': np.mean(timing_data['Prompt Augmentation']),
        'LLM Generation': np.mean(timing_data['LLM Generation'])
    }
    
    bottleneck = max(component_times.items(), key=lambda x: x[1])
    print(f"Primary bottleneck: {bottleneck[0]}")
    print(f"  Average time: {bottleneck[1]:.4f}s ({(bottleneck[1]/avg_total)*100:.2f}% of total)")
    
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
    print(f"\nComponents ranked by time:")
    for i, (comp, t) in enumerate(sorted_components, 1):
        print(f"  {i}. {comp}: {t:.4f}s ({(t/avg_total)*100:.2f}%)")
    
    print("="*80)
    
    # Throughput analysis
    print("\n" + "="*80)
    print("THROUGHPUT ANALYSIS")
    print("="*80)
    
    total_time = sum(timing_data['Total'])
    num_queries = len(results)
    throughput = num_queries / total_time
    
    print(f"Total queries: {num_queries}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average latency: {avg_total:.4f}s per query")
    print(f"Throughput: {throughput:.4f} queries/second")
    print(f"Throughput: {throughput * 60:.2f} queries/minute")
    print("="*80)
    
    return timing_data

def plot_timing_distributions(timing_data, output_file="timing_distributions.png"):
    """Create visualization of timing distributions."""
    print(f"\nGenerating timing distribution plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RAG Pipeline Component Timing Distributions', fontsize=16, fontweight='bold')
    
    components = ['Query Encoding', 'Vector Search', 'Document Retrieval', 
                  'Prompt Augmentation', 'LLM Generation', 'Total']
    
    for idx, (ax, component) in enumerate(zip(axes.flat, components)):
        times = timing_data[component]
        
        # Histogram
        ax.hist(times, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.4f}s')
        ax.axvline(np.median(times), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(times):.4f}s')
        
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(component, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")
    
    return output_file

def plot_component_breakdown(timing_data, output_file="component_breakdown.png"):
    """Create pie chart and bar chart of component breakdown."""
    print(f"Generating component breakdown plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('RAG Pipeline Component Time Breakdown', fontsize=16, fontweight='bold')
    
    components = ['Query Encoding', 'Vector Search', 'Document Retrieval', 
                  'Prompt Augmentation', 'LLM Generation']
    
    avg_times = [np.mean(timing_data[comp]) for comp in components]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # Pie chart
    ax1.pie(avg_times, labels=components, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Percentage of Total Time', fontsize=12, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(range(len(components)), avg_times, color=colors, edgecolor='black')
    ax2.set_xlabel('Component', fontsize=10)
    ax2.set_ylabel('Average Time (seconds)', fontsize=10)
    ax2.set_title('Average Time by Component', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels([c.replace(' ', '\n') for c in components], fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")
    
    return output_file

def save_results_to_json(results, timing_data, output_file="benchmark_results.json"):
    """Save benchmark results to JSON file."""
    print(f"\nSaving results to {output_file}...")
    
    output = {
        'num_queries': len(results),
        'statistics': {},
        'individual_queries': []
    }
    
    # Calculate statistics
    for component, times in timing_data.items():
        output['statistics'][component] = {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times))
        }
    
    # Save individual query results
    for result in results:
        output['individual_queries'].append({
            'query': result['query'],
            'timings': result['timings'],
            'num_retrieved_docs': len(result['retrieved_docs']),
            'response_length': len(result['response'])
        })
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")

def main():
    """Main function for Part 3 analysis."""
    print("\n" + "="*80)
    print("CS4414 HW3 - PART 3: SYSTEM ANALYSIS AND PERFORMANCE BENCHMARKING")
    print("="*80)
    
    # Configuration
    preprocessed_docs = "preprocessed_documents.json"
    llm_model = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    queries_file = "queries.json"
    num_queries = 20  # Number of queries to benchmark
    
    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(
        preprocessed_docs=preprocessed_docs,
        llm_model_path=llm_model,
        top_k=3
    )
    
    # Load test queries
    print(f"\nLoading test queries from {queries_file}...")
    queries = load_test_queries(queries_file, num_queries=num_queries)
    print(f"✓ Loaded {len(queries)} queries")
    
    # Run benchmark
    results = run_benchmark(pipeline, queries, warmup=2)
    
    # Analyze results
    timing_data = analyze_results(results)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_timing_distributions(timing_data)
    plot_component_breakdown(timing_data)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results_to_json(results, timing_data)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - timing_distributions.png")
    print("  - component_breakdown.png")
    print("  - benchmark_results.json")
    print("\nYou can use these results for your Part 3 analysis report!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

