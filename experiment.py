import os
import subprocess
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from itertools import combinations


def create_directories():
    directories = [
        'results/real', 'results/virtual', 'datasets/real', 'datasets/virtual', 'bin', 'results', 'logs'
    ]
    types = ['bruteForce', 'MinHash', 'LSHbase', 'bucketing', 'forest']
    for directory in directories:
        if 'results/' in directory:
            for t in types:
                os.makedirs(os.path.join(directory, t), exist_ok=True)
        else:
            os.makedirs(directory, exist_ok=True)


# Set up logging
def setup_logging():
    logging.basicConfig(filename='logs/experiment.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')


def run_corpus_mode(executable_path,
                    dataset_path,
                    output_dir,
                    k=None,
                    t=None,
                    b=None,
                    thr=None):
    """Run corpus mode experiment"""
    cmd = [executable_path, dataset_path]
    
    # Generate base filename parts based on provided parameters
    param_parts = []
    if k is not None:
        param_parts.append(f"k{k}")
    if b is not None:
        param_parts.append(f"b{b}")
    if t is not None:
        param_parts.append(f"t{t}")
    if thr is not None:
        param_parts.append(f"threshold{thr}")
    
    # Determine algorithm type from executable path
    if 'bruteForce' in executable_path:
        algo_type = 'bruteForce'
    elif 'MinHash' in executable_path:
        algo_type = 'MinHash'
    elif 'LSHbase' in executable_path:
        algo_type = 'LSHbase'
    elif 'LSHbucketing' in executable_path:
        algo_type = 'bucketing'
    elif 'LSHforest' in executable_path:
        algo_type = 'forest'
    else:
        algo_type = 'unknown'

    # Handle specific parameter requirements for LSH bucketing and forest
    if 'lsh_bucketing' in executable_path or 'lsh_forest' in executable_path:
        if k is None or t is None or b is None or thr is None:
            logging.error(
                f"Error: {executable_path} requires k, t, b, and thr parameters."
            )
            return {
                'dataset': dataset_path,
                'output': f"Error: {executable_path} requires k, t, b, and thr parameters.",
                'runtime': None,
                'status': 'error'
            }
        # Add parameters in the required order
        cmd.extend([str(k), str(b)])
    else:
        # Add parameters if provided for other executables
        if k is not None:
            cmd.append(str(k))
        if t is not None:
            cmd.append(str(t))
        if b is not None:
            cmd.append(str(b))
        if thr is not None:
            cmd.append(str(thr))

    # Create output file paths for both similarities and times
    similarity_csv = os.path.join(output_dir, f"{algo_type}Similarities_{'_'.join(param_parts)}.csv")
    times_csv = os.path.join(output_dir, f"{algo_type}Times_{'_'.join(param_parts)}.csv")

    try:
        start_time = time.time()
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        # Log the run
        logging.info(
            f"Successfully ran corpus mode {executable_path} on {dataset_path}"
        )

        return {
            'dataset': dataset_path,
            'similarity_csv': similarity_csv,
            'times_csv': times_csv,
            'runtime': end_time - start_time,
            'status': 'success',
            'method': algo_type,
            'k': k,
            't': t,
            'b': b,
            'thr': thr
        }
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Error running corpus mode {executable_path}: {e}"
        )
        return {
            'dataset': dataset_path,
            'output': e.stderr,
            'runtime': None,
            'status': 'error',
            'method': algo_type,
            'k': k,
            't': t,
            'b': b,
            'thr': thr
        }


def parse_csv_results(result):
    """Parse results from CSV files instead of output"""
    if result['status'] == 'error':
        return {
            'dataset': result['dataset'],
            'similar_pairs': [],
            'index_build_time': None,
            'query_time': None,
            'total_runtime': result['runtime'],
            'status': result['status'],
            'method': result['method'],
            'k': result['k'],
            't': result['t'],
            'b': result['b'],
            'thr': result['thr']
        }

    similar_pairs = []
    index_build_time = None
    query_time = None

    # Parse similarity CSV if it exists
    try:
        if os.path.exists(result['similarity_csv']):
            similarity_df = pd.read_csv(result['similarity_csv'])
            # Extract document pairs
            if not similarity_df.empty:
                # Assuming the CSV has columns for doc1, doc2, and similarity
                for _, row in similarity_df.iterrows():
                    # Adjust column names based on actual CSV format
                    doc1 = row[0] if len(row) > 0 else None
                    doc2 = row[1] if len(row) > 1 else None
                    if doc1 is not None and doc2 is not None:
                        similar_pairs.append((str(doc1), str(doc2)))
    except Exception as e:
        logging.error(f"Error parsing similarity CSV: {e}")

    # Parse times CSV if it exists
    try:
        if os.path.exists(result['times_csv']):
            times_df = pd.read_csv(result['times_csv'])
            # Extract timing information
            if not times_df.empty:
                # Assuming the CSV has columns for task and time
                for _, row in times_df.iterrows():
                    task = row[0] if len(row) > 0 else ""
                    time_value = row[1] if len(row) > 1 else None
                    
                    if isinstance(task, str):
                        if "index build" in task.lower() and time_value is not None:
                            index_build_time = float(time_value)
                        elif "query" in task.lower() and time_value is not None:
                            query_time = float(time_value)
    except Exception as e:
        logging.error(f"Error parsing times CSV: {e}")

    return {
        'dataset': result['dataset'],
        'similar_pairs': similar_pairs,
        'index_build_time': index_build_time,
        'query_time': query_time,
        'total_runtime': result['runtime'],
        'status': result['status'],
        'method': result['method'],
        'k': result['k'],
        't': result['t'],
        'b': result['b'],
        'thr': result['thr']
    }


def run_corpus_experiment(bin, dataset_dir, output_dir, k_values,
                          t_values, b_values, thr_values):
    """Run corpus mode experiments"""
    results = []

    # For each executable
    for exec_name, exec_path in bin.items():
        logging.info(f"Running corpus mode for {exec_name}")

        # Filter parameter values based on algorithm type
        b_values_filtered = b_values if 'lsh' in exec_name else [None]
        thr_values_filtered = thr_values if exec_name not in [
            'minhash', 'lsh_basic', 'brute_force'
        ] else [None]
        t_values_filtered = t_values if exec_name != 'brute_force' else [None]

        # For each parameter combination
        for k in k_values:
            for t in t_values_filtered:
                for b in b_values_filtered:
                    for thr in thr_values_filtered:
                        # Run the executable
                        result = run_corpus_mode(exec_path, dataset_dir,
                                                output_dir, k, t, b, thr)

                        # Parse the CSV results
                        parsed_result = parse_csv_results(result)

                        # For CSV export, convert similar_pairs to count
                        parsed_result['similar_pairs_count'] = len(
                            parsed_result['similar_pairs'])

                        results.append(parsed_result)

    # Create a DataFrame with the results
    df = pd.DataFrame(results)

    return df


def visualize_corpus_results(results_df, output_dir):
    """Create visualizations for corpus experiment results"""
    # Plot index build time vs k
    plt.figure(figsize=(12, 8))

    for method, group in results_df.groupby('method'):
        k_values = []
        build_times = []

        for k, k_group in group.groupby('k'):
            k_values.append(k)
            # Add error checking for missing values
            build_time = k_group['index_build_time'].mean()
            if pd.notna(build_time):
                build_times.append(build_time)
            else:
                build_times.append(0)  # or some placeholder value

        if k_values and build_times:  # Only plot if we have data
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
        if 't' in group.columns:  # Check if t exists in the dataframe
            t_values = []
            query_times = []

            for t, t_group in group.groupby('t'):
                if pd.notna(t):  # Skip None values
                    t_values.append(t)
                    # Add error checking for missing values
                    query_time = t_group['query_time'].mean()
                    if pd.notna(query_time):
                        query_times.append(query_time)
                    else:
                        query_times.append(0)  # or some placeholder value

            if t_values and query_times:  # Only plot if we have data
                plt.plot(t_values, query_times, marker='o', label=method)

    plt.title("Average Query Time vs. t")
    plt.xlabel("MinHash size (t)")
    plt.ylabel("Query Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "query_time_vs_t.png"))

    # Add a new plot for b parameter (LSH-specific)
    plt.figure(figsize=(12, 8))

    for method, group in results_df.groupby('method'):
        if 'lsh' in method and 'b' in group.columns:
            b_values = []
            query_times = []

            for b, b_group in group.groupby('b'):
                if pd.notna(b):
                    b_values.append(b)
                    query_time = b_group['query_time'].mean()
                    if pd.notna(query_time):
                        query_times.append(query_time)
                    else:
                        query_times.append(0)

            if b_values and query_times:
                plt.plot(b_values, query_times, marker='o', label=method)

    plt.title("Average Query Time vs. b (LSH parameter)")
    plt.xlabel("Number of bands (b)")
    plt.ylabel("Query Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "query_time_vs_b.png"))

    # Add a new plot for thr parameter
    plt.figure(figsize=(12, 8))

    for method, group in results_df.groupby('method'):
        if 'thr' in group.columns:
            thr_values = []
            similar_pairs_counts = []

            for thr, th_group in group.groupby('thr'):
                if pd.notna(thr):
                    thr_values.append(thr)
                    # Count the number of similar pairs
                    similar_pairs_count = th_group['similar_pairs_count'].mean(
                    )
                    similar_pairs_counts.append(similar_pairs_count)

            if thr_values and similar_pairs_counts:
                plt.plot(thr_values,
                         similar_pairs_counts,
                         marker='o',
                         label=method)

    plt.title("Average Number of Similar Pairs vs. thr")
    plt.xlabel("Similarity thr")
    plt.ylabel("Number of Similar Pairs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "similar_pairs_vs_thr.png"))


def generate_report(corpus_results, output_dir):
    """Generate a summary report"""
    report_path = os.path.join(output_dir, "experiment_report.txt")

    with open(report_path, 'w') as f:
        f.write("Document Similarity Methods Evaluation Report\n")
        f.write("===========================================\n\n")

        # Corpus Experiments
        if corpus_results is not None:
            f.write("Corpus Mode Results\n")
            f.write("---------------------\n\n")

            # Method comparison
            f.write("Method Performance Comparison:\n")

            # Safe way to find fastest/slowest methods
            method_build_times = {}
            method_query_times = {}

            for method, group in corpus_results.groupby('method'):
                build_times = group['index_build_time'].dropna()
                query_times = group['query_time'].dropna()

                if not build_times.empty:
                    avg_build_time = build_times.mean()
                    method_build_times[method] = avg_build_time
                    f.write(f"  {method}:\n")
                    f.write(
                        f"    Average index build time = {avg_build_time:.6f} seconds\n"
                    )

                if not query_times.empty:
                    avg_query_time = query_times.mean()
                    method_query_times[method] = avg_query_time
                    f.write(
                        f"    Average query time = {avg_query_time:.6f} seconds\n"
                    )

            f.write("\n")

            # Parameter effects
            f.write("Parameter Effects:\n")

            for method, group in corpus_results.groupby('method'):
                f.write(f"  {method}:\n")

                for k, k_group in group.groupby('k'):
                    build_times = k_group['index_build_time'].dropna()
                    if not build_times.empty:
                        avg_build_time = build_times.mean()
                        f.write(
                            f"    k={k}: Average build time = {avg_build_time:.6f} seconds\n"
                        )

                if 't' in group.columns:
                    for t, t_group in group.groupby('t'):
                        if pd.notna(t):
                            query_times = t_group['query_time'].dropna()
                            if not query_times.empty:
                                avg_query_time = query_times.mean()
                                f.write(
                                    f"    t={t}: Average query time = {avg_query_time:.6f} seconds\n"
                                )

                if 'b' in group.columns and 'lsh' in method:
                    for b, b_group in group.groupby('b'):
                        if pd.notna(b):
                            query_times = b_group['query_time'].dropna()
                            if not query_times.empty:
                                avg_query_time = query_times.mean()
                                f.write(
                                    f"    b={b}: Average query time = {avg_query_time:.6f} seconds\n"
                                )

                if 'thr' in group.columns:
                    for thr, th_group in group.groupby('thr'):
                        if pd.notna(thr):
                            similar_pairs_counts = th_group[
                                'similar_pairs_count']
                            if not similar_pairs_counts.empty:
                                avg_similar_pairs = similar_pairs_counts.mean()
                                f.write(
                                    f"    thr={thr}: Average similar pairs found = {avg_similar_pairs:.2f}\n"
                                )

                f.write("\n")

        # Observations and Recommendations
        f.write("Observations and Recommendations\n")
        f.write("---------------------------------\n\n")

        # Add observations based on results
        f.write("Observations:\n")

        if corpus_results is not None:
            # Compare methods by build time (safely)
            if method_build_times:
                fastest_build = min(method_build_times.items(),
                                    key=lambda x: x[1])[0]
                slowest_build = max(method_build_times.items(),
                                    key=lambda x: x[1])[0]
                f.write(
                    f"  - {fastest_build} has the fastest index build time\n")
                f.write(
                    f"  - {slowest_build} has the slowest index build time\n")

            # Compare methods by query time (safely)
            if method_query_times:
                fastest_query = min(method_query_times.items(),
                                    key=lambda x: x[1])[0]
                slowest_query = max(method_query_times.items(),
                                    key=lambda x: x[1])[0]
                f.write(f"  - {fastest_query} has the fastest query time\n")
                f.write(f"  - {slowest_query} has the slowest query time\n")

            # Analyze parameter effects
            f.write("  - Increasing k generally increases index build time\n")
            f.write("  - Increasing t affects query time and accuracy\n")
            if any('lsh' in method
                   for method in corpus_results['method'].unique()):
                f.write(
                    "  - LSH parameters (b, thr) offer a trade-off between speed and accuracy\n"
                )

        f.write("\n")

        f.write("Recommendations:\n")
        f.write(
            "  - For small document collections, brute force may be sufficient\n"
        )
        f.write(
            "  - For large collections, LSH methods offer better scaling\n")
        f.write(
            "  - Tune k based on document length and vocabulary diversity\n")
        f.write("  - Adjust t to balance accuracy and performance\n")
        if any('lsh' in method
               for method in corpus_results['method'].unique()):
            f.write(
                "  - For LSH, increase b for better accuracy at the cost of performance\n"
            )
            f.write("  - Choose thr based on the specific application needs\n")
        f.write("\n")

        # Conclusion
        f.write("Conclusion\n")
        f.write("-------------\n\n")
        f.write(
            "  The experiments provide a comprehensive evaluation of different Jaccard similarity\n"
        )
        f.write(
            "  computation methods. The choice of method and parameters should be based on the\n"
        )
        f.write(
            "  specific requirements of the application, particularly the trade-off between\n"
        )
        f.write("  accuracy and performance.\n")

    return report_path


def prepare_datasets(mode, num_docs):
    """Prepare datasets based on mode"""
    logging.info(f"Preparing {mode} datasets with {num_docs} documents...")

    if mode == 'real':
        gen_k = None
        cmd = ["./bin/exp1_genRandPerm", str(num_docs)]
    else:  # virtual mode
        gen_k = str(random.randint(4, 10))
        cmd = ["./bin/exp2_genRandShingles", gen_k, str(num_docs)]

    try:
        start_time = time.time()
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True)
        end_time = time.time()

        # Get the directory path from stdout
        # TODO: si quereis que no se imprima todos los archivos creados del exp1 y exp2 :: output_dir = result.stdout.strip()

        # Log the run
        logging.info(
            f"Successfully ran dataset generation and created dataset containing {num_docs} documents"
        )

        return {
            "output": result.stdout,
            "runtime": end_time - start_time,
            "status": "success",
        }
    except subprocess.CalledProcessError as e:
        logging.error(f"Error preparing datasets: {e}")
        return {
            "dataset": None,
            "output": e.stderr,
            "runtime": None,
            "status": "error",
        }


def main():
    parser = argparse.ArgumentParser(
        description='Document Similarity Methods Evaluation')
    parser.add_argument('--mode',
                        choices=['real', 'virtual'],
                        required=True,
                        help='Dataset mode: real or virtual')
    parser.add_argument('--k_values',
                        nargs='+',
                        type=int,
                        help='List of k values to test',
                        default=[3, 5, 7])
    parser.add_argument('--t_values',
                        nargs='+',
                        type=int,
                        help='List of t values to test',
                        default=[10, 100, 1000])
    parser.add_argument('--b_values',
                        nargs='+',
                        type=int,
                        help='List of b values to test',
                        default=[5, 10, 20, 50, 100])
    parser.add_argument('--thr_values',
                        nargs='+',
                        type=float,
                        default=[0.5, 0.6, 0.7, 0.8, 0.9],
                        help='List of thr values to test')
    parser.add_argument('--num_docs',
                        type=int,
                        default=20,
                        help='Number of documents to generate')
    parser.add_argument('--prepare_datasets',
                        action='store_true',
                        help='Prepare datasets before running experiments')

    args = parser.parse_args()

    create_directories()
    setup_logging()

    if args.prepare_datasets:
        result = prepare_datasets(args.mode, args.num_docs)
        if result['status'] == 'error':
            logging.error("Failed to prepare datasets. Exiting.")
            return

    bin = {
        'brute_force': './bin/jaccardBruteForce',
        'minhash': './bin/jaccardMinHash',
        'lsh_basic': './bin/jaccardLSHbase',
        'lsh_bucketing': './bin/jaccardLSHbucketing',
        'lsh_forest': './bin/jaccardLSHforest'
    }

    dataset_dir = os.path.join('datasets', args.mode)
    output_dir = os.path.join('results', args.mode, 'corpus')
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Running corpus experiments...")
    corpus_results = run_corpus_experiment(bin, dataset_dir,
                                           output_dir, args.k_values,
                                           args.t_values, args.b_values,
                                           args.thr_values)

    visualize_corpus_results(corpus_results, output_dir)
    generate_report(corpus_results, output_dir)

    logging.info("Experiments completed successfully.")


if __name__ == "__main__":
    main()