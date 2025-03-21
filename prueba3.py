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
        'datasets/real', 'datasets/virtual', 'executables', 'results', 'logs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Set up logging
def setup_logging():
    logging.basicConfig(filename='logs/experiment.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')


def run_corpus_mode(executable_path,
                    dataset_path,
                    output_file,
                    k=None,
                    t=None,
                    b=None,
                    threshold=None):
    """Run corpus mode experiment"""
    cmd = [executable_path, dataset_path]

    # Handle specific parameter requirements for LSH bucketing and forest
    if 'lsh_bucketing' in executable_path or 'lsh_forest' in executable_path:
        if k is None or t is None or b is None or threshold is None:
            logging.error(
                f"Error: {executable_path} requires k, t, b, and threshold parameters."
            )
            return {
                'dataset': dataset_path,
                'output':
                f"Error: {executable_path} requires k, t, b, and threshold parameters.",
                'runtime': None,
                'status': 'error'
            }
        # Add parameters in the required order
        cmd.extend([str(k), str(b), str(t), str(threshold)])
    else:
        # Add parameters if provided for other executables
        if k is not None:
            cmd.append(str(k))
        if t is not None:
            cmd.append(str(t))
        if b is not None:
            cmd.append(str(b))
        if threshold is not None:
            cmd.append(str(threshold))

    # Fix: Don't include redirection in the command list
    command_str = " ".join(cmd) + " > " + output_file

    try:
        start_time = time.time()
        # Fix: Use shell=True to handle redirection
        result = subprocess.run(command_str,
                                shell=True,
                                capture_output=True,
                                text=True,
                                check=True)
        end_time = time.time()

        # Read the output from the file instead
        try:
            with open(output_file, 'r') as f:
                output = f.read()
        except FileNotFoundError:
            output = ""
            logging.warning(f"Output file {output_file} not found")

        # Log the run
        logging.info(
            f"Successfully ran corpus mode {executable_path} on {dataset_path}"
        )

        return {
            'dataset': dataset_path,
            'output': output,
            'runtime': end_time - start_time,
            'status': 'success'
        }
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Error running corpus mode {executable_path}: {e} \n try running: {command_str}"
        )
        return {
            'dataset': dataset_path,
            'output': e.stderr,
            'runtime': None,
            'status': 'error'
        }


def parse_corpus_output(result):
    """Parse the output from a corpus mode experiment"""
    # This function will need to be adapted based on the actual output format of your C++ executables
    if not result['output']:
        return {
            'dataset': result['dataset'],
            'similar_pairs': [],
            'index_build_time': None,
            'query_time': None,
            'total_runtime': result['runtime'],
            'status': result['status']
        }

    lines = result['output'].strip().split('\n')

    # Example parsing - adjust based on your actual output format
    similar_pairs = []
    index_build_time = None
    query_time = None

    if 'lsh_bucketing' in result['dataset']:
        for line in lines:
            if line.startswith("time:"):
                # Extract the total runtime from the line
                try:
                    total_runtime = int(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    total_runtime = None
            elif "," in line:
                # Parse the line containing the similar pairs
                parts = line.split(",")
                if len(parts) >= 4:
                    doc1 = parts[0].strip()
                    doc2 = parts[1].strip()
                    est_similarity = float(parts[2].strip())
                    exact_similarity = float(parts[3].strip())
                    similar_pairs.append(
                        (doc1, doc2, est_similarity, exact_similarity))
    else:

        for line in lines:
            if 'Similar pair:' in line:
                pair = line.split(':')[1].strip()
                doc1, doc2 = pair.split(',')
                similar_pairs.append((doc1.strip(), doc2.strip()))
            elif 'Index build time:' in line:
                try:
                    index_build_time = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'Query time:' in line:
                try:
                    query_time = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass

        return {
            'dataset': result['dataset'],
            'similar_pairs': similar_pairs,
            'index_build_time': index_build_time,
            'query_time': query_time,
            'total_runtime': result['runtime'],
            'status': result['status']
        }


def run_corpus_experiment(executables, dataset_dir, output_dir, k_values,
                          t_values, b_values, threshold_values):
    """Run corpus mode experiments"""
    results = []

    # For each executable
    for exec_name, exec_path in executables.items():
        logging.info(f"Running corpus mode for {exec_name}")

        # Filter parameter values based on algorithm type
        b_values_filtered = b_values if 'lsh' in exec_name else [None]
        threshold_values_filtered = threshold_values if exec_name not in [
            'minhash', 'lsh_basic', 'brute_force'
        ] else [None]
        t_values_filtered = t_values if exec_name != 'brute_force' else [None]

        # For each parameter combination
        for k in k_values:
            for t in t_values_filtered:
                for b in b_values_filtered:
                    for threshold in threshold_values_filtered:
                        output_file = os.path.join(
                            output_dir,
                            f"{exec_name}_corpus_k{k}_t{t or 'NA'}_b{b or 'NA'}_th{threshold or 'NA'}.csv"
                        )

                        # Run the executable
                        result = run_corpus_mode(exec_path, dataset_dir,
                                                 output_file, k, t, b,
                                                 threshold)

                        # Parse the output
                        parsed_result = parse_corpus_output(result)

                        # Add method and parameter info
                        parsed_result['method'] = exec_name
                        parsed_result['k'] = k
                        parsed_result['t'] = t
                        parsed_result['b'] = b
                        parsed_result['threshold'] = threshold

                        # For CSV export, convert similar_pairs to count
                        parsed_result['similar_pairs_count'] = len(
                            parsed_result['similar_pairs'])

                        results.append(parsed_result)

    # Create a DataFrame with the results
    df = pd.DataFrame(results)

    # Reformat the DataFrame for better CSV export
    csv_df = pd.DataFrame({
        'method': df['method'],
        'k': df['k'],
        't': df['t'],
        'b': df['b'],
        'threshold': df['threshold'],
        'similar_pairs_count': df['similar_pairs_count'],
        'index_build_time': df['index_build_time'],
        'query_time': df['query_time'],
        'total_runtime': df['total_runtime'],
        'status': df['status']
    })

    # Save all results
    csv_df.to_csv(os.path.join(output_dir, "corpus_all_results.csv"),
                  index=False)

    # Save summary results by method
    summary_by_method = csv_df.groupby('method').agg({
        'index_build_time': ['mean', 'min', 'max'],
        'query_time': ['mean', 'min', 'max'],
        'similar_pairs_count': ['mean', 'min', 'max'],
        'total_runtime': ['mean', 'min', 'max']
    }).reset_index()
    summary_by_method.columns = [
        '_'.join(col).strip('_') for col in summary_by_method.columns.values
    ]
    summary_by_method.to_csv(os.path.join(output_dir, "summary_by_method.csv"),
                             index=False)

    # Save summary by k parameter
    summary_by_k = csv_df.groupby(['method', 'k']).agg({
        'index_build_time': 'mean',
        'query_time': 'mean',
        'similar_pairs_count': 'mean',
        'total_runtime': 'mean'
    }).reset_index()
    summary_by_k.to_csv(os.path.join(output_dir, "summary_by_k.csv"),
                        index=False)

    # Save summary by t parameter (for applicable methods)
    t_methods = csv_df[csv_df['t'].notna()]
    if not t_methods.empty:
        summary_by_t = t_methods.groupby(['method', 't']).agg({
            'index_build_time':
            'mean',
            'query_time':
            'mean',
            'similar_pairs_count':
            'mean',
            'total_runtime':
            'mean'
        }).reset_index()
        summary_by_t.to_csv(os.path.join(output_dir, "summary_by_t.csv"),
                            index=False)

    # Save summary by b parameter (for LSH methods)
    b_methods = csv_df[csv_df['b'].notna()]
    if not b_methods.empty:
        summary_by_b = b_methods.groupby(['method', 'b']).agg({
            'index_build_time':
            'mean',
            'query_time':
            'mean',
            'similar_pairs_count':
            'mean',
            'total_runtime':
            'mean'
        }).reset_index()
        summary_by_b.to_csv(os.path.join(output_dir, "summary_by_b.csv"),
                            index=False)

    # Save summary by threshold (for applicable methods)
    threshold_methods = csv_df[csv_df['threshold'].notna()]
    if not threshold_methods.empty:
        summary_by_threshold = threshold_methods.groupby(
            ['method', 'threshold']).agg({
                'index_build_time': 'mean',
                'query_time': 'mean',
                'similar_pairs_count': 'mean',
                'total_runtime': 'mean'
            }).reset_index()
        summary_by_threshold.to_csv(os.path.join(output_dir,
                                                 "summary_by_threshold.csv"),
                                    index=False)

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

    # Add a new plot for threshold parameter
    plt.figure(figsize=(12, 8))

    for method, group in results_df.groupby('method'):
        if 'threshold' in group.columns:
            threshold_values = []
            similar_pairs_counts = []

            for threshold, th_group in group.groupby('threshold'):
                if pd.notna(threshold):
                    threshold_values.append(threshold)
                    # Count the number of similar pairs
                    similar_pairs_count = th_group['similar_pairs_count'].mean(
                    )
                    similar_pairs_counts.append(similar_pairs_count)

            if threshold_values and similar_pairs_counts:
                plt.plot(threshold_values,
                         similar_pairs_counts,
                         marker='o',
                         label=method)

    plt.title("Average Number of Similar Pairs vs. Threshold")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Number of Similar Pairs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "similar_pairs_vs_threshold.png"))


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

                if 'threshold' in group.columns:
                    for threshold, th_group in group.groupby('threshold'):
                        if pd.notna(threshold):
                            similar_pairs_counts = th_group[
                                'similar_pairs_count']
                            if not similar_pairs_counts.empty:
                                avg_similar_pairs = similar_pairs_counts.mean()
                                f.write(
                                    f"    threshold={threshold}: Average similar pairs found = {avg_similar_pairs:.2f}\n"
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
                    "  - LSH parameters (b, threshold) offer a trade-off between speed and accuracy\n"
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
            f.write(
                "  - Choose threshold based on the specific application needs\n"
            )
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
        cmd = ["./executables/exp1_genRandPerm", str(num_docs)]
    else:  # virtual mode
        gen_k = str(random.randint(4, 10))
        cmd = ["./executables/exp2_genRandShingles", gen_k, str(num_docs)]

    try:
        start_time = time.time()
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True)
        end_time = time.time()

        # Get the directory path from stdout
        output_dir = result.stdout.strip()

        # Log the run
        logging.info(
            f"Successfully ran dataset generation and created {output_dir} containing {num_docs} documents"
        )

        return {
            "dataset": output_dir,
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
    parser.add_argument('--threshold_values',
                        nargs='+',
                        type=float,
                        default=[0.5, 0.6, 0.7, 0.8, 0.9],
                        help='List of threshold values to test')
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

    executables = {
        'brute_force': './executables/jaccardBruteForce',
        'minhash': './executables/jaccardMinHash',
        'lsh_basic': './executables/jaccardLSHbase',
        'lsh_bucketing': './executables/jaccardLSHbucketing',
        'lsh_forest': './executables/jaccardLSHforest'
    }

    dataset_dir = os.path.join('datasets', args.mode)
    output_dir = os.path.join('results', args.mode, 'corpus')
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Running corpus experiments...")
    corpus_results = run_corpus_experiment(executables, dataset_dir,
                                           output_dir, args.k_values,
                                           args.t_values, args.b_values,
                                           args.threshold_values)

    visualize_corpus_results(corpus_results, output_dir)
    generate_report(corpus_results, output_dir)

    logging.info("Experiments completed successfully.")


if __name__ == "__main__":
    main()
