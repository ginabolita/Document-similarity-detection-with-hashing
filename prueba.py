import os
import subprocess
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import time
import copy
from itertools import combinations

# Create directory structure
def create_directories():
    directories = ['datasets/real', 'datasets/virtual', 'executables', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
# Set up logging
def setup_logging():
    logging.basicConfig(
        filename='logs/experiment.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

'''
#----------------------------------------------------
# GENERACIÓN DOCUMENTO BASE
#----------------------------------------------------
def generate_base_document(min_distinct_words, output_path):
    """Generate a base document with at least min_distinct_words distinct words"""
    # Simple implementation - in practice, you might want to use real text
    words = set()
    with open(output_path, 'w') as f:
        while len(words) < min_distinct_words:
            #FIXME: mirar k valor random
            #TODO: deberiamos eliminar stopwords antes
            #TODO: marcel: mejor que sacara las palabras del json o algo
            word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))
            words.add(word)
            f.write(word + ' ')
    return output_path

#----------------------------------------------------
# EXPERIMENTOS GENERACIÓN DOCUMENTOS
#----------------------------------------------------
# permutaciones del documento base a nivel de palabra
def create_permuted_documents(base_doc_path, num_docs, output_dir):
    """Create permuted versions of the base document"""
    with open(base_doc_path, 'r') as f:
        content = f.read()
    
    words = content.split()
    
    for i in range(num_docs):
        # Create a permuted version by shuffling some words
        permuted_words = words.copy()
        
        # Modify approximately 10-30% of the document
        #TODO: revisar intervalo a modificar, actualmente está en 10-30, ver si es mejor más rango
        num_changes = random.randint(int(len(words) * 0.1), int(len(words) * 0.3))
        indices_to_change = random.sample(range(len(words)), num_changes)
        
        for idx in indices_to_change:
            # Either replace with a new word, swap with another, or remove
            action = random.choice(['replace', 'swap', 'remove'])
            
            if action == 'replace':
                permuted_words[idx] = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))
            elif action == 'swap' and len(permuted_words) > 1:
                swap_idx = random.randint(0, len(permuted_words) - 1)
                permuted_words[idx], permuted_words[swap_idx] = permuted_words[swap_idx], permuted_words[idx]
            elif action == 'remove':
                permuted_words[idx] = ''
        
        # Write the permuted document
        output_path = os.path.join(output_dir, f'doc_{i+1}.txt')
        with open(output_path, 'w') as f:
            f.write(' '.join(word for word in permuted_words if word))

def create_virtual_documents(base_doc_path, num_docs, k, output_dir, similarity_range=(0.3, 0.9)):
    """Create virtual documents with controlled similarity by sampling k-shingles"""
    with open(base_doc_path, 'r') as f:
        content = f.read()
    
    # Generate k-shingles from the base document
    shingles = []
    for i in range(len(content) - k + 1):
        shingles.append(content[i:i+k])
    
    # Create virtual documents with controlled similarity
    similarities = []
    for i in range(num_docs):
        # Choose a target similarity
        #FIXME: aquí la target similarity no será random
        target_similarity = random.uniform(similarity_range[0], similarity_range[1])
        
        # Calculate how many shingles to sample
        num_shingles = int(len(shingles) * target_similarity)
        
        # Sample shingles
        sampled_shingles = random.sample(shingles, num_shingles)
        
        # Create the virtual document
        virtual_doc = ''.join(sampled_shingles)
        
        # Write the virtual document
        output_path = os.path.join(output_dir, f'virtual_{i+1}.txt')
        with open(output_path, 'w') as f:
            f.write(virtual_doc)
        
        similarities.append((f'virtual_{i+1}.txt', target_similarity))
    
    # Save the intended similarities for later comparison
    with open(os.path.join(output_dir, 'intended_similarities.txt'), 'w') as f:
        for doc, sim in similarities:
            f.write(f"{doc}\t{sim}\n")

#----------------------------------------------------
# EXPERIMENTOS TESTING
#----------------------------------------------------
def run_one_on_one(executable_path, doc1_path, doc2_path, output_file, k=None, t=None):
    """Run a one-on-one comparison between two documents"""
    cmd = [executable_path, doc1_path, doc2_path, output_file]
    
    # Add k and t parameters if provided
    if k is not None:
        cmd.append(str(k))
    if t is not None:
        cmd.append(str(t))
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        # Parse the output
        stdout = result.stdout
        
        # Log the run
        logging.info(f"Successfully ran {executable_path} on {doc1_path} and {doc2_path}")
        
        return {
            'doc1': os.path.basename(doc1_path),
            'doc2': os.path.basename(doc2_path),
            'output': stdout,
            'runtime': end_time - start_time,
            'status': 'success'
        }
    except subprocess.CalledProcessError as e:
        #print(f"error running {executable_path}: {e}")
        logging.error(f"Error running {executable_path}: {e}")
        return {
            'doc1': os.path.basename(doc1_path),
            'doc2': os.path.basename(doc2_path),
            'output': e.stderr,
            'runtime': None,
            'status': 'error'
        }
'''

def run_corpus_mode(executable_path, dataset_path, output_file, k=None, t=None, b=None, threshold=None):
    """Run corpus mode experiment"""
    cmd = [executable_path, dataset_path]
    
    # Add parameters if provided
    if k is not None:
        cmd.append(str(k))
    
    if t is not None:
        cmd.append(str(t))

    if b is not None:
        cmd.append(str(b))

    if threshold is not None:
        cmd.append(str(threshold))
    
    cmd.extend([">", output_file])
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        # Parse the output
        stdout = result.stdout
        
        # Log the run
        logging.info(f"Successfully ran corpus mode {executable_path} on {dataset_path}")
        
        return {
            'dataset': dataset_path,
            'output': stdout,
            'runtime': end_time - start_time,
            'status': 'success'
        }
    except subprocess.CalledProcessError as e:
        command = " ".join(cmd)
        #print(f"error running {executable_path}: {e}, try running: {command} ")
        logging.error(f"Error running corpus mode {executable_path}: {e} \n try running: {command}")
        return {
            'dataset': dataset_path,
            'output': e.stderr,
            'runtime': None,
            'status': 'error'
        }

#----------------------------------------------------
# EXPERIMENTOS PARSING
#----------------------------------------------------
def parse_one_on_one_output(result):
    """Parse the output from a one-on-one comparison"""
    # This function will need to be adapted based on the actual output format of your C++ executables
    lines = result['output'].strip().split('\n')
    
    # Example parsing - adjust based on your actual output format
    similarity = None
    hash_count = None
    
    for line in lines:
        if 'Similarity:' in line:
            similarity = float(line.split(':')[1].strip())
        elif 'Hash count:' in line:
            hash_count = int(line.split(':')[1].strip())
    
    return {
        'doc1': result['doc1'],
        'doc2': result['doc2'],
        'similarity': similarity,
        'hash_count': hash_count,
        'runtime': result['runtime'],
        'status': result['status']
    }

def parse_corpus_output(result):
    """Parse the output from a corpus mode experiment"""
    # This function will need to be adapted based on the actual output format of your C++ executables
    lines = result['output'].strip().split('\n')
    
    # Example parsing - adjust based on your actual output format
    similar_pairs = []
    index_build_time = None
    query_time = None
    
    for line in lines:
        if 'Similar pair:' in line:
            pair = line.split(':')[1].strip()
            doc1, doc2 = pair.split(',')
            similar_pairs.append((doc1.strip(), doc2.strip()))
        elif 'Index build time:' in line:
            index_build_time = float(line.split(':')[1].strip())
        elif 'Query time:' in line:
            query_time = float(line.split(':')[1].strip())
    
    return {
        'dataset': result['dataset'],
        'similar_pairs': similar_pairs,
        'index_build_time': index_build_time,
        'query_time': query_time,
        'total_runtime': result['runtime'],
        'status': result['status']
    }

#----------------------------------------------------
# EXPERIMENTOS
#----------------------------------------------------
def run_one_on_one_experiment(executables, dataset_dir, output_dir, k_values, t_values, b_values):
    """Run one-on-one comparisons for multiple executables"""
    # Get list of documents
    docs = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.txt')])
    
    results = []
    
    # Generate all document pairs
    doc_pairs = list(combinations(docs, 2))
    
    # For each executable
    for exec_name, exec_path in executables.items():
        logging.info(f"Running {exec_name} on {len(doc_pairs)} document pairs")
        
        # For each parameter combination
        for k in k_values:
            for t in t_values:
                # Skip t parameter for brute force
                if exec_name == 'brute_force' and t is not None:
                    continue
                
                k_param = k if exec_name != 'brute_force' else None
                t_param = t if exec_name != 'brute_force' else None
                
                param_results = []
                
                # For each document pair
                for doc1, doc2 in doc_pairs:
                    doc1_path = os.path.join(dataset_dir, doc1)
                    doc2_path = os.path.join(dataset_dir, doc2)
                    output_file = os.path.join(output_dir, f"{exec_name}_{doc1}_{doc2}_k{k}_t{t}.txt")
                    
                    # Run the executable
                    result = run_one_on_one(exec_path, doc1_path, doc2_path, output_file, k_param, t_param)
                    
                    # Parse the output
                    parsed_result = parse_one_on_one_output(result)
                    
                    # Add method and parameter info
                    parsed_result['method'] = exec_name
                    parsed_result['k'] = k
                    parsed_result['t'] = t
                    
                    param_results.append(parsed_result)
                
                # Save results for this parameter combination
                param_df = pd.DataFrame(param_results)
                param_df.to_csv(os.path.join(output_dir, f"{exec_name}_k{k}_t{t}_results.csv"), index=False)
                
                results.extend(param_results)
    
    # Save all results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "one_on_one_all_results.csv"), index=False)
    
    return df

def run_corpus_experiment(executables, dataset_dir, output_dir, k_values, t_values, b_values, threshold_values):
    """Run corpus mode experiments"""
    results = []
    
    # For each executable
    for exec_name, exec_path in executables.items():
        logging.info(f"Running corpus mode for {exec_name}")

        b_values_filtered = b_values if 'lsh' in exec_name else [None]
        threshold_values_filtered = threshold_values if exec_name not in ['minhash', 'lsh_basic', 'brute_force'] else [None]
        t_values_filtered = t_values if exec_name != 'brute_force' else [None]

        
        # For each parameter combination
        for k in k_values:
            for t in t_values_filtered:
                for b in b_values_filtered:
                    for threshold in threshold_values_filtered:
                        output_file = os.path.join(output_dir, f"{exec_name}_corpus_k{k}_t{t or 'NA'}_b{b or 'NA'}_th{threshold or 'NA'}.txt")
                        
                        # Run the executable
                        result = run_corpus_mode(exec_path, dataset_dir, output_file, k, t, b, threshold)
                        
                        # Parse the output
                        parsed_result = parse_corpus_output(result)
                        
                        # Add method and parameter info
                        parsed_result['method'] = exec_name
                        parsed_result['k'] = k
                        parsed_result['t'] = t
                        parsed_result['threshold'] = threshold
                        
                        results.append(parsed_result)
    
    # Save all results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "corpus_all_results.csv"), index=False)
    
    return df


#----------------------------------------------------
# EXPERIMENTOS visualización
#----------------------------------------------------
def visualize_one_on_one_results(results_df, output_dir):
    """Create visualizations for one-on-one experiment results"""
    # Group by method and k
    grouped = results_df.groupby(['method', 'k'])
    
    # Plot runtime vs k for each method
    plt.figure(figsize=(12, 8))
    
    for method, group in results_df.groupby('method'):
        k_values = []
        runtimes = []
        
        for k, k_group in group.groupby('k'):
            k_values.append(k)
            runtimes.append(k_group['runtime'].mean())
        
        plt.plot(k_values, runtimes, marker='o', label=method)
    
    plt.title("Average Runtime vs. k for Different Methods")
    plt.xlabel("Shingle size (k)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "runtime_vs_k.png"))
    
    # Plot accuracy vs k for MinHash and LSH
    # (using brute force as ground truth)
    plt.figure(figsize=(12, 8))
    
    # Get brute force similarities as ground truth
    brute_force_df = results_df[results_df['method'] == 'brute_force']
    brute_force_similarities = {}
    
    for _, row in brute_force_df.iterrows():
        pair_key = (row['doc1'], row['doc2'])
        brute_force_similarities[pair_key] = row['similarity']
    
    # Calculate error for other methods
    for method in ['minhash', 'lsh_basic']:
        method_df = results_df[results_df['method'] == method]
        k_values = []
        errors = []
        
        for k, k_group in method_df.groupby('k'):
            k_values.append(k)
            
            # Calculate average absolute error
            error_sum = 0
            count = 0
            
            for _, row in k_group.iterrows():
                pair_key = (row['doc1'], row['doc2'])
                if pair_key in brute_force_similarities:
                    error = abs(row['similarity'] - brute_force_similarities[pair_key])
                    error_sum += error
                    count += 1
            
            if count > 0:
                errors.append(error_sum / count)
            else:
                errors.append(0)
        
        plt.plot(k_values, errors, marker='o', label=method)
    
    plt.title("Average Error vs. k (Compared to Brute Force)")
    plt.xlabel("Shingle size (k)")
    plt.ylabel("Average Absolute Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_vs_k.png"))

def visualize_corpus_results(results_df, output_dir):
    """Create visualizations for corpus experiment results"""
    # Plot index build time vs k
    plt.figure(figsize=(12, 8))
    
    for method, group in results_df.groupby('method'):
        k_values = []
        build_times = []
        
        for k, k_group in group.groupby('k'):
            k_values.append(k)
            build_times.append(k_group['index_build_time'].mean())
        
        plt.plot(k_values, build_times, marker='o', label=method)
    
    plt.title("Average Index Build Time vs. k")
    plt.xlabel("Shingle size (k)")
    plt.ylabel("Build Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "build_time_vs_k.png"))
    
    # Plot query time vs t
    plt.figure(figsize=(12, 8))
    
    for method, group in results_df.groupby('method'):
        t_values = []
        query_times = []
        
        for t, t_group in group.groupby('t'):
            t_values.append(t)
            query_times.append(t_group['query_time'].mean())
        
        plt.plot(t_values, query_times, marker='o', label=method)
    
    plt.title("Average Query Time vs. t")
    plt.xlabel("MinHash size (t)")
    plt.ylabel("Query Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "query_time_vs_t.png"))

def generate_report(one_on_one_results, corpus_results, output_dir):
    """Generate a summary report"""
    report_path = os.path.join(output_dir, "experiment_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Document Similarity Methods Evaluation Report\n")
        f.write("===========================================\n\n")
        
        # One-on-One Experiments
        if one_on_one_results is not None:
            f.write("1. One-on-One Comparison Results\n")
            f.write("-------------------------------\n\n")
            
            # Method comparison
            f.write("Method Performance Comparison:\n")
            
            for method, group in one_on_one_results.groupby('method'):
                avg_runtime = group['runtime'].mean()
                f.write(f"  {method}: Average runtime = {avg_runtime:.6f} seconds\n")
            
            f.write("\n")
            
            # Parameter effects
            f.write("Parameter Effects:\n")
            
            for method, group in one_on_one_results.groupby('method'):
                if method != 'brute_force':
                    f.write(f"  {method}:\n")
                    
                    for k, k_group in group.groupby('k'):
                        avg_runtime = k_group['runtime'].mean()
                        f.write(f"    k={k}: Average runtime = {avg_runtime:.6f} seconds\n")
                    
                    f.write("\n")
            
            # Accuracy comparison
            if 'brute_force' in one_on_one_results['method'].unique():
                f.write("Accuracy Comparison (vs. Brute Force):\n")
                
                # Get brute force similarities as ground truth
                brute_force_df = one_on_one_results[one_on_one_results['method'] == 'brute_force']
                brute_force_similarities = {}
                
                for _, row in brute_force_df.iterrows():
                    pair_key = (row['doc1'], row['doc2'])
                    brute_force_similarities[pair_key] = row['similarity']
                
                # Calculate error for other methods
                for method in one_on_one_results['method'].unique():
                    if method != 'brute_force':
                        method_df = one_on_one_results[one_on_one_results['method'] == method]
                        
                        error_sum = 0
                        count = 0
                        
                        for _, row in method_df.iterrows():
                            pair_key = (row['doc1'], row['doc2'])
                            if pair_key in brute_force_similarities:
                                error = abs(row['similarity'] - brute_force_similarities[pair_key])
                                error_sum += error
                                count += 1
                        
                        if count > 0:
                            avg_error = error_sum / count
                            f.write(f"  {method}: Average absolute error = {avg_error:.6f}\n")
                
                f.write("\n")
        
        # Corpus Experiments
        if corpus_results is not None:
            f.write("2. Corpus Mode Results\n")
            f.write("---------------------\n\n")
            
            # Method comparison
            f.write("Method Performance Comparison:\n")
            
            for method, group in corpus_results.groupby('method'):
                avg_build_time = group['index_build_time'].mean()
                avg_query_time = group['query_time'].mean()
                f.write(f"  {method}:\n")
                f.write(f"    Average index build time = {avg_build_time:.6f} seconds\n")
                f.write(f"    Average query time = {avg_query_time:.6f} seconds\n")
            
            f.write("\n")
            
            # Parameter effects
            f.write("Parameter Effects:\n")
            
            for method, group in corpus_results.groupby('method'):
                f.write(f"  {method}:\n")
                
                for k, k_group in group.groupby('k'):
                    avg_build_time = k_group['index_build_time'].mean()
                    f.write(f"    k={k}: Average build time = {avg_build_time:.6f} seconds\n")
                
                for t, t_group in group.groupby('t'):
                    avg_query_time = t_group['query_time'].mean()
                    f.write(f"    t={t}: Average query time = {avg_query_time:.6f} seconds\n")
                
                f.write("\n")
        
        # Observations and Recommendations
        f.write("3. Observations and Recommendations\n")
        f.write("---------------------------------\n\n")
        
        # Add your observations here
        f.write("Observations:\n")
        f.write("  - [Add your observations based on the results]\n\n")
        
        f.write("Recommendations:\n")
        f.write("  - [Add your recommendations based on the results]\n\n")
        
        # Conclusion
        f.write("4. Conclusion\n")
        f.write("-------------\n\n")
        f.write("  - [Add your conclusion based on the experimental results]\n")
    
    return report_path
#----------------------------------------------------
# MAIN FUNCTION
#----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Document Similarity Methods Evaluation')
    parser.add_argument('--mode', choices=['real', 'virtual'], required=True, help='Dataset mode: real or virtual')
    parser.add_argument('--experiment', choices=['one-on-one', 'corpus'], required=True, help='Experiment type: one-on-one or corpus')
    parser.add_argument('--k_values', nargs='+', type=int, help='List of k values to test', default=[3, 5, 7])
    parser.add_argument('--t_values', nargs='+', type=int, help='List of t values to test', default=[10, 100, 1000])
    parser.add_argument('--b_values', nargs='+', type=int, help='List of t values to test', default=[5, 10, 20, 50, 100])
    parser.add_argument('--threshold_values', nargs='+', type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9], help='List of threshold values to test (for corpus mode)')
    parser.add_argument('--num_docs', type=int, default=20, help='Number of documents to generate')
    parser.add_argument('--prepare_datasets', action='store_true', help='Prepare datasets before running experiments')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Setup logging
    setup_logging()
    
    # Prepare datasets if requested
    if args.prepare_datasets:
        logging.info("Preparing datasets...")
        if args.mode == 'real':
            gen_k = None
            cmd = "./executables/exp1_genRandPerm"
        else:
            gen_k = str(random.randint(4, 10))
            cmd = "./executables/exp2_genRandShingles"
        try:
            start_time = time.time()
            result = subprocess.run(
                [cmd, gen_k, str(args.num_docs)], capture_output=True, text=True, check=True
            )
            end_time = time.time()

            # Get the directory path from stdout
            output_dir = result.stdout.strip()

            # Log the run
            logging.info(f"Successfully ran exp1 and created {output_dir} containing {args.num_docs} documents")

            return {
                "dataset": output_dir,
                "output": result.stdout,
                "runtime": end_time - start_time,
                "status": "success",
            }
        except subprocess.CalledProcessError as e:
            #print(f"Error running: {e}")
            logging.error(f"Error running exp1: {e}")
            return {
                "dataset": None,
                "output": e.stderr,
                "runtime": None,
                "status": "error",
            }
    
    # Define executable paths
    executables = {
        'brute_force': './executables/jaccardBruteForce',
        'minhash': './executables/jaccardMinHash',
        'lsh_basic': './executables/jaccardLSHbase',
        'lsh_bucketing': './executables/jaccardLSHbucketing',
        'lsh_forest': './executables/jaccardLSHforest'
    }
    
    # Run experiments
    dataset_dir = os.path.join('datasets', args.mode)
    output_dir = os.path.join('results', args.mode, args.experiment)
    os.makedirs(output_dir, exist_ok=True)
    
    one_on_one_results = None
    corpus_results = None

    if args.experiment == 'one-on-one':
        logging.info("Running one-on-one experiments...")
        one_on_one_results = run_one_on_one_experiment(executables, dataset_dir, output_dir, args.k_values, args.t_values, args.b_values)
        visualize_one_on_one_results(one_on_one_results, output_dir)
    else:  # corpus mode
        logging.info("Running corpus experiments...")
        #print(executables, dataset_dir, output_dir, args.k_values, args.t_values, args.b_values, args.threshold_values)
        corpus_results = run_corpus_experiment(executables, dataset_dir, output_dir, args.k_values, args.t_values, args.b_values, args.threshold_values)
        visualize_corpus_results(corpus_results, output_dir)

    # Generate report using the appropriate variables
    generate_report(one_on_one_results, corpus_results, output_dir)
        
    logging.info("Experiments completed successfully.")

if __name__ == "__main__":
    main()