import os
import subprocess
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
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


def run_corpus_mode(executable_path, dataset_path, output_dir, k=None, t=None, b=None, thr=None, run_idx=None):
    """Run corpus mode experiment"""
    cmd = [executable_path, dataset_path]
    
    # Generate base filename parts based on provided parameters
    param_parts = []
    if k is not None:
        param_parts.append(f"k{k}")
    if t is not None:
        param_parts.append(f"t{t}")
    if b is not None:
        param_parts.append(f"b{b}")
    
    # Determine algorithm type from executable path
    if 'BruteForce' in executable_path:
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
    if 'LSHbucketing' in executable_path or 'LSHforest' in executable_path:
        if k is None or t is None or b is None or thr is None:
            logging.error(f"Error: {executable_path} requires k, t, b, and thr parameters.")
            return {
                'dataset': dataset_path,
                'output': f"Error: {executable_path} requires k, t, b, and thr parameters.",
                'runtime': None,
                'status': 'error'
            }

    # Add parameters if provided for other executables
    if k is not None:
        cmd.append(str(k))
    if t is not None:
        cmd.append(str(t))
    if b is not None:
        cmd.append(str(b))
    if thr is not None and algo_type in ['bucketing', 'forest']:
        cmd.append(str(thr))

    # Only append threshold to filename for bucketing and forest
    if algo_type in ['bucketing', 'forest'] and thr is not None:
        param_parts.append(f"threshold{thr}")

    try:
        start_time = time.time()
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        logging.info(f"Successfully ran corpus mode {executable_path} on {dataset_path}")

        # Define default output CSV file paths
        similarity_csv_default = os.path.join(output_dir, f"{algo_type}/{algo_type}Similarities_{'_'.join(param_parts)}.csv")
        times_csv_default = os.path.join(output_dir, f"{algo_type}/{algo_type}Times_{'_'.join(param_parts)}.csv")

        # Rename files to include run_idx if provided
        if run_idx is not None:
            similarity_csv = similarity_csv_default.replace('.csv', f'_run{run_idx}.csv')
            times_csv = times_csv_default.replace('.csv', f'_run{run_idx}.csv')
            
            if os.path.exists(similarity_csv_default):
                os.rename(similarity_csv_default, similarity_csv)
            if os.path.exists(times_csv_default):
                os.rename(times_csv_default, times_csv)
        else:
            similarity_csv = similarity_csv_default
            times_csv = times_csv_default

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
            'thr': thr,
            'run_idx': run_idx
        }
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running corpus mode {executable_path}: {e}")
        return {
            'dataset': dataset_path,
            'output': e.stderr,
            'runtime': None,
            'status': 'error',
            'method': algo_type,
            'k': k,
            't': t,
            'b': b,
            'thr': thr,
            'run_idx': run_idx
        }

def parse_csv_results(result):
    """Parse results from CSV files instead of output"""
    if result['status'] == 'error':
        return {
            'dataset': result['dataset'],
            'similarity_pairs': [],
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

    similarity_pairs = []
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
                    doc1 = row["Doc1"] if len(row) > 0 else None
                    doc2 = row["Doc2"] if len(row) > 1 else None
                    similarity = row["Sim%"] if len(row) > 2 else None
                    threshold = result['thr'] if result['thr'] is not None else None
                    if threshold is None:
                        print("=============== WARNING:  Threshold is None")
                    if doc1 is not None and doc2 is not None and similarity is not None and similarity > threshold:
                        similarity_pairs.append((str(doc1), str(doc2), similarity))
            else:
                logging.error(f"No similar pairs found for {result['method']}")
        else:
            logging.error(f"Similarity CSV not found for {result['method']}")
    except Exception as e:
        logging.error(f"Error parsing similarity CSV: {e}")
        print(f"Times CSV not found for: {result['similarity_csv']}")

    # Parse times CSV if it exists
    try:
        if os.path.exists(result['times_csv']):
            times_df = pd.read_csv(result['times_csv'])
            # Extract timing information
            if not times_df.empty:
                # Assuming the CSV has columns for task and time
                for _, row in times_df.iterrows():
                    task = row["Operation"] if len(row) > 0 else ""
                    time_value = row["Time(ms)"] if len(row) > 1 else None
                    
                    if isinstance(task, str):
                        if "index build" in task.lower() and time_value is not None:
                            index_build_time = float(time_value)
                        elif "query" in task.lower() and time_value is not None:
                            query_time = float(time_value)
                        elif "time" in task.lower() and time_value is not None:
                            total_runtime = float(time_value)
                        else:
                            logging.error(f"Invalid task name in times CSV: {task}")
                            print(f"Invalid task name in times CSV: {task}")
                    else:
                        logging.error(f"Invalid task name in times CSV: {task}")
            else:
                logging.error(f"No timing information found for {result['method']}")
        else:
            logging.error(f"Times CSV not found for {result['method']}")
            print(f"Times CSV not found for: {result['times_csv']}")
    except Exception as e:
        logging.error(f"Error parsing times CSV: {e}")

    return {
        'dataset': result['dataset'],
        'similarity_pairs': similarity_pairs,
        'index_build_time': index_build_time,
        'query_time': query_time,
        'total_runtime': total_runtime,
        'status': result['status'],
        'method': result['method'],
        'k': result['k'],
        't': result['t'],
        'b': result['b'],
        'thr': result['thr'],
        'similarity_csv': result['similarity_csv'],
        'times_csv': result['times_csv']
    }


def run_parameter_experiment(bin, dataset_dir, output_dir, param_to_vary, 
                            base_k=5, base_t=500, base_b=50, base_thr=0.3, num_runs=30):
    """Run experiments varying one parameter while fixing others, repeating each num_runs times"""
    results = []
    
    base_b_value = int(base_t * (base_b / 100.0)) if base_b is not None else None
    
    logging.info(f"Running experiment varying {param_to_vary} with {num_runs} runs per configuration")
    
    for exec_name, exec_path in bin.items():
        logging.info(f"Running {exec_name} with varying {param_to_vary}")
        
        uses_t = exec_name != 'brute_force'
        uses_b = 'lsh' in exec_name
        uses_thr = exec_name not in ['minhash', 'lsh_basic', 'brute_force']
        
        k_val = base_k
        t_val = base_t if uses_t else None
        b_val = base_b_value if uses_b else None
        thr_val = base_thr
        
        if param_to_vary == 'k':
            values_to_try = list(range(1, 15))
        elif param_to_vary == 't' and uses_t:
            values_to_try = list(range(100, 1001, 100))
        elif param_to_vary == 'b' and uses_b:
            values_to_try = [int(base_t * (pct / 100.0)) for pct in range(10, 101, 10)]
        elif param_to_vary == 'thr' and uses_thr:
            values_to_try = [round(x * 0.1, 1) for x in range(1, 10)]
        else:
            values_to_try = []
            
        for val in values_to_try:
            if param_to_vary == 'k':
                k_val = val
            elif param_to_vary == 't':
                t_val = val
                if uses_b:
                    b_val = int(val * (base_b / 100.0))
            elif param_to_vary == 'b':
                b_val = val
            elif param_to_vary == 'thr':
                thr_val = val
                
            # Run the experiment num_runs times
            for run_idx in range(1, num_runs + 1):
                result = run_corpus_mode(exec_path, dataset_dir, output_dir, 
                                        k_val, t_val, b_val, thr_val, run_idx)
                
                parsed_result = parse_csv_results(result)
                parsed_result['similarity_pairs_count'] = len(parsed_result['similarity_pairs'])
                parsed_result['varied_param'] = param_to_vary
                parsed_result['varied_value'] = val
                parsed_result['run_idx'] = run_idx
                
                results.append(parsed_result)
    
    df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, f"results_vary_{param_to_vary}.csv")
    df.to_csv(results_file, index=False)
    return df

def prepare_datasets(mode, num_docs):
    """Prepare datasets based on mode"""
    logging.info(f"Preparing {mode} datasets with {num_docs} documents...")

    # if datasets already exist, erase them
    if len(os.listdir(f'datasets/{mode}/')) > 0:
        logging.info("Deleting existing datasets...")
        for file in os.listdir(f'datasets/{mode}/'):
            os.remove(f'datasets/{mode}/{file}')

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
    

def plot_parameter_comparison(results_df, param_name, metric_name, output_dir):
    """Plot performance comparison with means and std error bars for varying parameter values"""
    plt.figure(figsize=(10, 6))
    
    algorithms = results_df['method'].unique()
    
    for algo in algorithms:
        algo_df = results_df[results_df['method'] == algo]
        if not algo_df.empty:
            # Group by varied_value to compute mean and std
            grouped = algo_df.groupby('varied_value')[metric_name].agg(['mean', 'std']).reset_index()
            plt.errorbar(grouped['varied_value'], grouped['mean'], yerr=grouped['std'], 
                         marker='o', label=algo, capsize=5)
    
    param_labels = {
        'k': 'Shingle Size (k)',
        't': 'Number of Hash Functions (t)',
        'b': 'Number of Bands (b)',
        'thr': 'Similarity Threshold'
    }
    metric_labels = {
        'total_runtime': 'Total Runtime (ms)',
        'index_build_time': 'Index Build Time (ms)',
        'query_time': 'Query Time (ms)',
        'similarity_pairs_count': 'Number of Similar Pairs'
    }
    
    plt.xlabel(param_labels.get(param_name, param_name))
    plt.ylabel(metric_labels.get(metric_name, metric_name))
    plt.title(f'Effect of {param_labels.get(param_name, param_name)} on {metric_labels.get(metric_name, metric_name)}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_file = os.path.join(output_dir, f'plot_{param_name}_{metric_name}.png')
    plt.savefig(plot_file)
    plt.close()
    
    logging.info(f"Plot saved to {plot_file}")
    return plot_file

def plot_algorithm_comparison(results_dfs, metric_name, output_dir):
    plt.figure(figsize=(12, 8))
    combined_df = pd.concat(results_dfs, ignore_index=True)
    
    # Create a copy of the dataframe to identify and filter outliers
    filtered_df = combined_df.copy()
    
    # Filter outliers for each method group
    methods = filtered_df['method'].unique()
    for method in methods:
        method_data = filtered_df.loc[filtered_df['method'] == method, metric_name]
        q1 = method_data.quantile(0.25)
        q3 = method_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Create a mask for non-outliers and apply it
        outlier_mask = (method_data < lower_bound) | (method_data > upper_bound)
        filtered_df.loc[(filtered_df['method'] == method) & outlier_mask, metric_name] = np.nan
    
    # Remove rows with NaN values for plotting
    filtered_df = filtered_df.dropna(subset=[metric_name])
    
    # Boxplot with data points (filtered)
    sns.boxplot(x='method', y=metric_name, data=combined_df, showfliers=False)
    sns.stripplot(x='method', y=metric_name, data=filtered_df, color="black", size=4, jitter=True)
    
    metric_labels = {
        'total_runtime': 'Average Runtime (ms)',
        'index_build_time': 'Average Index Build Time (ms)',
        'query_time': 'Average Query Time (ms)',
        'similarity_pairs_count': 'Average Number of Similar Pairs'
    }
    
    plt.ylabel(metric_labels.get(metric_name, metric_name))
    plt.title(f'Algorithm Comparison: {metric_labels.get(metric_name, metric_name)}')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plot_file = os.path.join(output_dir, f'algorithm_comparison_{metric_name}.png')
    plt.savefig(plot_file)
    plt.close()
    
    logging.info(f"Plot saved to {plot_file}")
    return plot_file


# Improved get_precision function with better error handling
def get_precision(file_path1, file_path2):
    """
    Calculate the precision between two similarity CSV files by comparing
    the similarity values (third column).
    
    Args:
        file_path1: Path to the first CSV file (ground truth)
        file_path2: Path to the second CSV file (to compare)
        
    Returns:
        float: Average precision value between 0 and 1
    """
    try:
        # Load both CSV files
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
        
        # Get the column names
        cols1 = df1.columns.tolist()
        cols2 = df2.columns.tolist()
        
        # Verify we have at least 3 columns in each file
        if len(cols1) < 3 or len(cols2) < 3:
            logging.error(f"CSV files don't have enough columns: {file_path1}, {file_path2}")
            return None
            
        # Get the third column (similarity values)
        sim_col1 = df1.iloc[:, 2]
        sim_col2 = df2.iloc[:, 2]
        
        # Make sure the dataframes are sorted the same way
        # Sort by first two columns (presumably document IDs)
        df1 = df1.sort_values(by=[cols1[0], cols1[1]]).reset_index(drop=True)
        df2 = df2.sort_values(by=[cols2[0], cols2[1]]).reset_index(drop=True)
        
        sim_values1 = df1.iloc[:, 2].tolist()
        sim_values2 = df2.iloc[:, 2].tolist()
        
        # Handle length mismatch
        if len(sim_values1) != len(sim_values2):
            # Find document pairs present in both files
            # Create tuple keys from the first two columns (doc IDs)
            pairs1 = set(zip(df1.iloc[:, 0], df1.iloc[:, 1]))
            pairs2 = set(zip(df2.iloc[:, 0], df2.iloc[:, 1]))
            
            # Get common pairs
            common_pairs = pairs1.intersection(pairs2)
            
            if not common_pairs:
                logging.warning(f"No common document pairs between {file_path1} and {file_path2}")
                return None
                
            # Extract similarity values for common pairs
            sim_values1 = []
            sim_values2 = []
            
            for pair in common_pairs:
                idx1 = df1[(df1.iloc[:, 0] == pair[0]) & (df1.iloc[:, 1] == pair[1])].index[0]
                idx2 = df2[(df2.iloc[:, 0] == pair[0]) & (df2.iloc[:, 1] == pair[1])].index[0]
                
                sim_values1.append(df1.iloc[idx1, 2])
                sim_values2.append(df2.iloc[idx2, 2])
        
        # Calculate precision for each pair
        precision_values = [1 - abs(v1 - v2) for v1, v2 in zip(sim_values1, sim_values2)]
        
        # Calculate average precision
        avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0
        
        return avg_precision
        
    except Exception as e:
        logging.error(f"Error calculating precision between {file_path1} and {file_path2}: {str(e)}")
        return None
    
def compare_accuracy(results_dfs, output_dir):
    """
    Compare accuracy of algorithms against brute force (ground truth) and plot results
    as line plots with error bars.
    
    Args:
        results_dfs: List of DataFrames with results from different parameter experiments
        output_dir: Directory to save the plots
    """
    # Find all brute force results to use as ground truth
    brute_force_results = []
    for df in results_dfs:
        bf_df = df[df['method'] == 'bruteForce']
        if not bf_df.empty:
            brute_force_results.append(bf_df)
    
    if not brute_force_results:
        logging.error("No brute force results found for accuracy comparison")
        return
    
    # Combine all brute force results
    combined_bf = pd.concat(brute_force_results)
    
    # Get a set of all similar pairs found by brute force (ground truth)
    ground_truth_pairs = set()
    for pairs in combined_bf['similarity_pairs']:
        if isinstance(pairs, list):
            ground_truth_pairs.update(tuple(sorted((pair[0], pair[1]))) for pair in pairs)
    
    # Calculate precision and recall for each algorithm
    accuracy_results = []
    
    for df in results_dfs:
        for _, row in df.iterrows():
            if row['method'] != 'bruteForce':
                algo_pairs = set()
                if isinstance(row['similarity_pairs'], list):
                    algo_pairs = set(tuple(sorted((pair[0], pair[1]))) for pair in row['similarity_pairs'])
                
                # Calculate precision and recall if ground truth has pairs
                if ground_truth_pairs:
                    true_positives = len(algo_pairs.intersection(ground_truth_pairs))
                    precision = true_positives / len(algo_pairs) if algo_pairs else 0
                    recall = true_positives / len(ground_truth_pairs)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision, recall, f1 = 0, 0, 0
                
                accuracy_results.append({
                    'method': row['method'],
                    'varied_param': row['varied_param'],
                    'varied_value': row['varied_value'],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
    
    # Create DataFrame for plotting
    accuracy_df = pd.DataFrame(accuracy_results).reset_index(drop=True)
    
    # Define styles for each method to match plot_* aesthetics
    method_styles = {
        'MinHash': {'color': 'blue', 'marker': 'D'},    # Diamond
        'LSHbase': {'color': 'orange', 'marker': 's'},  # Square
        'bucketing': {'color': 'green', 'marker': 'o'}, # Circle
        'forest': {'color': 'red', 'marker': '^'}       # Triangle
    }
    
    # Generate plots for each varied parameter
    for param in accuracy_df['varied_param'].unique():
        param_df = accuracy_df[accuracy_df['varied_param'] == param]
        
        plt.figure(figsize=(10, 6))
        
        # Plot each method
        methods = param_df['method'].unique()
        for method in methods:
            style = method_styles.get(method, {'color': 'gray', 'marker': 'o'})  # Fallback style
            method_df = param_df[param_df['method'] == method]
            # Group by varied_value and compute mean and std of f1_score
            grouped = method_df.groupby('varied_value')['f1_score'].agg(['mean', 'std']).reset_index()
            plt.errorbar(
                grouped['varied_value'],
                grouped['mean'],
                yerr=grouped['std'],
                marker=style['marker'],
                color=style['color'],
                label=method,
                capsize=5  # Size of error bar caps
            )
        
        # Set plot labels and styling to match plot_parameter_comparison
        param_labels = {
            'k': 'Shingle Size (k)',
            't': 'Number of Hash Functions (t)',
            'b': 'Number of Bands (b)',
            'thr': 'Similarity Threshold'
        }
        
        plt.xlabel(param_labels.get(param, param))
        plt.ylabel('F1 Score')
        plt.title(f'F1 Score vs {param_labels.get(param, param)}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Method')  # Include legend with method title
        
        # Save the plot
        plot_file = os.path.join(output_dir, f'accuracy_{param}.png')
        plt.savefig(plot_file)
        plt.close()
        
        logging.info(f"Accuracy plot saved to {plot_file}")
    
    return accuracy_df


def compare_similarity_accuracy(results_dfs, output_dir, mode):
    """
    Compare similarity accuracy of algorithms against brute force ground truth.
    
    Args:
        results_dfs (list): List of DataFrames with experiment results.
        output_dir (str): Directory to save output CSV and plots.
        mode (str): 'real' or 'virtual' (not used in this implementation).
    
    Returns:
        pd.DataFrame or None: DataFrame with precision results or None if no results.
    """
    # Step 1: Collect all similarity CSV paths into a nested dictionary
    all_similarity_csvs = {}
    for df in results_dfs:
        for _, row in df.iterrows():
            if isinstance(row['similarity_csv'], str) and os.path.exists(row['similarity_csv']):
                method = row['method']
                param = row['varied_param']
                value = row['varied_value']
                run_idx = row['run_idx']
                all_similarity_csvs.setdefault(method, {}).setdefault(param, {}).setdefault(value, {})[run_idx] = row['similarity_csv']

    # Step 2: Initialize list to store precision results
    similarity_precision_results = []

    # Define the base k value used for brute force when varying other parameters
    base_k = 5  # Adjust this if your base k is different

    # Step 3: Iterate over each parameter type
    for param in ['k', 't', 'b', 'thr']:
        # Check if this parameter was varied in any DataFrame
        if not any(df['varied_param'].eq(param).any() for df in results_dfs):
            continue
        
        # Concatenate all rows for this parameter and get unique values
        param_df = pd.concat([df[df['varied_param'] == param] for df in results_dfs])
        values_tried = sorted(param_df['varied_value'].unique())
        if not values_tried:
            continue

        # Step 4: Process each varied value
        for value in values_tried:
            # Select brute force file as ground truth
            bf_file = None
            if param == 'k':
                # For 'k', use the brute force file for that specific k value, run 1
                if ('bruteForce' in all_similarity_csvs and 
                    'k' in all_similarity_csvs['bruteForce'] and 
                    value in all_similarity_csvs['bruteForce']['k']):
                    bf_file = all_similarity_csvs['bruteForce']['k'][value][1]  # Use run 1
            else:
                # For 't', 'b', 'thr', use the brute force file for the base k (e.g., k=5), run 1
                if ('bruteForce' in all_similarity_csvs and 
                    'k' in all_similarity_csvs['bruteForce'] and 
                    base_k in all_similarity_csvs['bruteForce']['k']):
                    bf_file = all_similarity_csvs['bruteForce']['k'][base_k][1]  # Use run 1

            if not bf_file or not os.path.exists(bf_file):
                logging.warning(f"No bruteForce file found for {param}={value} (base k={base_k})")
                continue

            # Step 5: Compare with approximate methods
            for algo in ['MinHash', 'LSHbase', 'bucketing', 'forest']:
                if (algo in all_similarity_csvs and 
                    param in all_similarity_csvs[algo] and 
                    value in all_similarity_csvs[algo][param]):
                    run_files = all_similarity_csvs[algo][param][value]
                    for run_idx, algo_file in run_files.items():
                        if os.path.exists(algo_file):
                            precision = get_precision(bf_file, algo_file)
                            if precision is not None:
                                similarity_precision_results.append({
                                    'method': algo,
                                    'varied_param': param,
                                    'varied_value': value,
                                    'run_idx': run_idx,
                                    'similarity_precision': precision
                                })

    # Step 6: Process results if any
    if similarity_precision_results:
        # Create DataFrame
        sim_precision_df = pd.DataFrame(similarity_precision_results)
        precision_csv = os.path.join(output_dir, 'similarity_precision_results.csv')
        sim_precision_df.to_csv(precision_csv, index=False)
        logging.info(f"Similarity precision results saved to {precision_csv}")

        # Step 7: Create boxplots for each parameter
        for param in ['k', 't', 'b', 'thr']:
            param_df = sim_precision_df[sim_precision_df['varied_param'] == param]
            if not param_df.empty:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='varied_value', y='similarity_precision', hue='method', data=param_df)
                param_labels = {
                    'k': 'Shingle Size (k)',
                    't': 'Number of Hash Functions (t)',
                    'b': 'Number of Bands (b)',
                    'thr': 'Similarity Threshold'
                }
                plt.title(f'Similarity Precision vs {param_labels.get(param, param)}')
                plt.xlabel(param_labels.get(param, param))
                plt.ylabel('Similarity Precision')
                plt.legend(title='Method')
                plt.grid(True, linestyle='--', alpha=0.7)
                plot_file = os.path.join(output_dir, f'similarity_precision_{param}.png')
                plt.savefig(plot_file)
                plt.close()
                logging.info(f"Boxplot saved to {plot_file}")

        # Step 8: Create overall boxplot by method
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='method', y='similarity_precision', data=sim_precision_df)
        plt.title('Similarity Precision by Method')
        plt.xlabel('Method')
        plt.ylabel('Similarity Precision')
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_file = os.path.join(output_dir, 'similarity_precision_by_method.png')
        plt.savefig(plot_file)
        plt.close()
        logging.info(f"Boxplot saved to {plot_file}")

        return sim_precision_df
    else:
        logging.warning("No similarity precision results generated.")
    return None


def create_heatmap(csv_file, output_dir):
    """
    Create a heatmap from the similarity CSV file.
    
    Args:
        csv_file: Path to the CSV file containing similarity results.
        output_dir: Directory to save the heatmap image.
    """
    try:
        # Leer el archivo CSV
        df = pd.read_csv(csv_file)
        
        # Verificar si tiene las columnas necesarias
        if 'Doc1' not in df.columns or 'Doc2' not in df.columns or 'Sim%' not in df.columns:
            logging.error(f"El archivo CSV {csv_file} no tiene las columnas requeridas (Doc1, Doc2, Sim%)")
            return
            
        # Extraer información del nombre del archivo para el nombre del heatmap
        base_filename = os.path.basename(csv_file)
        # Eliminar la extensión
        base_name = os.path.splitext(base_filename)[0]
        # Extraer el tipo de algoritmo (parte antes de "Similarities")
        algo_type = base_name.split('Similarities')[0] if 'Similarities' in base_name else 'unknown'
        # Extraer los parámetros si existen (parte después de "Similarities_")
        params_part = base_name.split('Similarities_')[1] if 'Similarities_' in base_name else ''
        
        # Crear un nombre único para el heatmap
        heatmap_name = f'heatmap_{algo_type}{params_part}.png'
        
        # Crear la matriz de similitud
        # Asegurarse de que Doc1 y Doc2 sean tratados como strings para la creación del pivot
        df['Doc1'] = df['Doc1'].astype(str)
        df['Doc2'] = df['Doc2'].astype(str)
        
        # Crear la matriz pivot
        similarity_matrix = pd.pivot_table(df, values='Sim%', index='Doc1', columns='Doc2', fill_value=0)
        
        # Determinar el valor máximo de la matriz
        max_value = similarity_matrix.max().max()  # Valor máximo en toda la matriz
        
        # Configurar el heatmap en función del valor máximo
        if max_value <= 0.1:
            # Si el valor máximo es 0.1 o menor, usar una paleta azul y vmax=0.1
            cmap = 'gist_heat'
            vmax = 0.1
        else:
            # De lo contrario, usar la paleta RdYlGn y vmax=1
            cmap = 'RdYlGn'
            vmax = 1
        
        # Crear el heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=False,  # Cambiar a True si quieres valores en las casillas
            fmt=".2f",
            cmap=cmap,    # Paleta de colores (azul o RdYlGn)
            cbar=True,
            vmin=0,       # Límite inferior de la escala
            vmax=vmax     # Límite superior de la escala (0.1 o 1)
        )
        
        # Configurar el título y las etiquetas
        plt.title(f'Heatmap de Similitud entre Documentos - {algo_type}{" " + params_part if params_part else ""}')
        plt.xlabel('Doc2')
        plt.ylabel('Doc1')
        
        # Guardar el heatmap
        heatmap_file = os.path.join(output_dir, heatmap_name)
        plt.savefig(heatmap_file)
        plt.close()
        
        logging.info(f"Heatmap guardado en {heatmap_file}")
        return heatmap_file
    except Exception as e:
        logging.error(f"Error al crear el heatmap: {e}")
        return None

def analyze_and_visualize_results(mode, experiment_types=['vary_k', 'vary_t', 'vary_b', 'vary_thr']):
    """
    Analyze results from all experiments and generate visualization
    
    Args:
        mode: Dataset mode ('real' or 'virtual')
        experiment_types: List of experiment types to analyze
    """
    logging.info(f"Analyzing and visualizing results for {mode} mode")
    
    output_dir = os.path.join('results', mode)
    results_dfs = []
    
    # Load result CSVs
    for exp_type in experiment_types:
        param = exp_type.split('_')[1]
        results_file = os.path.join(output_dir, f"results_vary_{param}.csv")
        
        if os.path.exists(results_file):
            try:
                df = pd.read_csv(results_file).reset_index(drop=True)
                
                # Convert similarity_pairs from string to actual lists if needed
                if 'similarity_pairs' in df.columns and df['similarity_pairs'].dtype == 'object':
                    df['similarity_pairs'] = df['similarity_pairs'].apply(eval)
                
                results_dfs.append(df)
                logging.info(f"Loaded results from {results_file}")
            except Exception as e:
                logging.error(f"Error loading results from {results_file}: {e}")
    
    if not results_dfs:
        logging.error("No result files found for analysis")
        return
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate runtime comparison plots for each parameter variation
    metrics = ['total_runtime', 'index_build_time', 'query_time', 'similarity_pairs_count']
    
    for df in results_dfs:
        if not df.empty:
            param_name = df['varied_param'].iloc[0]
            
            for metric in metrics:
                if metric in df.columns:
                    plot_parameter_comparison(df, param_name, metric, viz_dir)
    
    # Generate algorithm comparison plots
    for metric in metrics:
        plot_algorithm_comparison(results_dfs, metric, viz_dir)
    
    # Compare accuracy using both approaches
    accuracy_df = compare_accuracy(results_dfs, viz_dir)
    
    # Use the improved similarity precision comparison (friend's method, enhanced)
    similarity_precision_df = compare_similarity_accuracy(results_dfs, viz_dir, mode)
    
    # Create heatmaps from similarity CSV files
    for algo_type in ['bruteForce', 'MinHash', 'LSHbase', 'bucketing', 'forest']:
        similarity_dir = os.path.join(output_dir, algo_type)
        if os.path.exists(similarity_dir):
            # Find similarity CSV files
            similarity_files = [f for f in os.listdir(similarity_dir) if f.endswith('.csv') and 'Similarities' in f]
            for sim_file in similarity_files:
                sim_file_path = os.path.join(similarity_dir, sim_file)
                # only create heatmap for the first run
                # check if the file is a first run file
                if sim_file.endswith('1.csv'):
                    create_heatmap(sim_file_path, viz_dir)
    
    # Generate a comprehensive summary report
    create_summary_report(results_dfs, accuracy_df, similarity_precision_df, output_dir)
    
    logging.info(f"Analysis and visualization completed. Reports saved to {output_dir}")

def create_summary_report(results_dfs, accuracy_df, similarity_precision_df, output_dir):
    combined_df = pd.concat(results_dfs)
    
    method_summary = combined_df.groupby('method').agg({
        'total_runtime': ['mean', 'std'],
        'index_build_time': ['mean', 'std'],
        'query_time': ['mean', 'std'],
        'similarity_pairs_count': ['mean', 'std']
    }).reset_index()
    method_summary.columns = ['method', 'runtime_mean', 'runtime_std', 'index_mean', 'index_std', 
                             'query_mean', 'query_std', 'pairs_mean', 'pairs_std']
    
    fastest_method = method_summary.loc[method_summary['runtime_mean'].idxmin()]['method']
    non_bf_summary = method_summary[method_summary['method'] != 'bruteForce']
    most_pairs_method = non_bf_summary.loc[non_bf_summary['pairs_mean'].idxmax()]['method'] if not non_bf_summary.empty else None
    
    if accuracy_df is not None and not accuracy_df.empty:
        avg_accuracy = accuracy_df.groupby('method')['f1_score'].mean().reset_index()
        most_accurate_method_f1 = avg_accuracy.loc[avg_accuracy['f1_score'].idxmax()]['method']
    else:
        most_accurate_method_f1 = None
    
    if similarity_precision_df is not None and not similarity_precision_df.empty:
        avg_precision = similarity_precision_df.groupby('method')['similarity_precision'].mean().reset_index()
        most_accurate_method_sim = avg_precision.loc[avg_precision['similarity_precision'].idxmax()]['method']
    else:
        most_accurate_method_sim = None
    
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("# Document Similarity Methods Evaluation Summary\n\n")
        f.write("## Overall Performance\n")
        f.write(f"- Fastest method: {fastest_method}\n")
        if most_pairs_method:
            f.write(f"- Method finding most similar pairs: {most_pairs_method}\n")
        if most_accurate_method_f1:
            f.write(f"- Most accurate method (F1 score): {most_accurate_method_f1}\n")
        if most_accurate_method_sim:
            f.write(f"- Most accurate method (Similarity precision): {most_accurate_method_sim}\n")
        
        f.write("\n## Method Comparison (Mean ± Std)\n")
        method_summary['runtime'] = method_summary.apply(
            lambda row: f"{row['runtime_mean']:.2f} ± {row['runtime_std']:.2f}", axis=1)
        method_summary['pairs'] = method_summary.apply(
            lambda row: f"{row['pairs_mean']:.1f} ± {row['pairs_std']:.1f}", axis=1)
        f.write(method_summary[['method', 'runtime', 'pairs']].to_string(index=False))
        
        # Add accuracy comparison if available
        if accuracy_df is not None and not accuracy_df.empty:
            f.write("\n\n## Document Pair Accuracy (F1 Score)\n")
            avg_f1 = accuracy_df.groupby('method')['f1_score'].mean().reset_index()
            avg_f1['f1_score'] = avg_f1['f1_score'].map('{:.3f}'.format)
            f.write(avg_f1.to_string(index=False))
            
        # Add similarity precision comparison if available
        if similarity_precision_df is not None and not similarity_precision_df.empty:
            f.write("\n\n## Similarity Value Precision\n")
            avg_sim = similarity_precision_df.groupby('method')['similarity_precision'].mean().reset_index()
            avg_sim['similarity_precision'] = avg_sim['similarity_precision'].map('{:.3f}'.format)
            f.write(avg_sim.to_string(index=False))
        
        f.write("\n\n## Parameter Effects\n")
        for df in results_dfs:
            if not df.empty:
                param = df['varied_param'].iloc[0]
                f.write(f"\n### Effect of varying {param}\n")
                
                # Find optimal parameter value for each method
                for method in df['method'].unique():
                    method_df = df[df['method'] == method]
                    if not method_df.empty:
                        best_runtime_idx = method_df['total_runtime'].idxmin()
                        best_value = method_df.loc[best_runtime_idx]['varied_value']
                        f.write(f"- Best {param} value for {method} (runtime): {best_value}\n")
                        
                # Check if we have accuracy data for this parameter
                if accuracy_df is not None and not accuracy_df.empty:
                    param_accuracy = accuracy_df[accuracy_df['varied_param'] == param]
                    if not param_accuracy.empty:
                        f.write(f"\n#### Best {param} values for F1 score:\n")
                        for method in param_accuracy['method'].unique():
                            method_acc = param_accuracy[param_accuracy['method'] == method]
                            if not method_acc.empty:
                                best_acc_idx = method_acc['f1_score'].idxmax()
                                best_value = method_acc.loc[best_acc_idx]['varied_value']
                                f.write(f"- Best {param} value for {method} (F1 score): {best_value}\n")
                                
                # Check if we have similarity precision data for this parameter
                if similarity_precision_df is not None and not similarity_precision_df.empty:
                    param_precision = similarity_precision_df[similarity_precision_df['varied_param'] == param]
                    if not param_precision.empty:
                        f.write(f"\n#### Best {param} values for similarity precision:\n")
                        for method in param_precision['method'].unique():
                            method_prec = param_precision[param_precision['method'] == method]
                            if not method_prec.empty:
                                best_prec_idx = method_prec['similarity_precision'].idxmax()
                                best_value = method_prec.loc[best_prec_idx]['varied_value']
                                f.write(f"- Best {param} value for {method} (similarity precision): {best_value}\n")
    
    logging.info(f"Summary report created at {os.path.join(output_dir, 'summary_report.txt')}")


def get_third_column_values(file_path):
    try:
        df = pd.read_csv(file_path)  
        third_column = df.iloc[:, 2].tolist()  
        return third_column
    except Exception as e:
        print(f"Error: {e}")
        return []

def precisions_files_var(csvs, char, values_to_try, mode):
    allpaths = csvs['similarity_csv']
    bruteForce = sorted([path for path in allpaths if 'bruteForceSimilarities' in path])
    LSHbase = sorted([path for path in allpaths if 'LSHbaseSimilarities' in path])
    MinHash = sorted([path for path in allpaths if 'MinHashSimilarities' in path])
    
    PrecisionMinHash = []
    PrecisionLSHbase = []
    
    for k in values_to_try:
        if char == 'k':
            bruteForceK = [path for path in bruteForce if f'k{k}_run' in path]
            MinHashK = [path for path in MinHash if f'k{k}_run' in path]
            LSHbaseK = [path for path in LSHbase if f'k{k}_run' in path]
        elif char == 't':
            # Find bruteForce files with base k and any run index
            if mode == "real":
                brute_force_dir = 'results/real/bruteForce/'
            else:
                brute_force_dir = 'results/virtual/bruteForce/'
            bruteForceK = [os.path.join(brute_force_dir, f) for f in os.listdir(brute_force_dir)
                           if f.startswith('bruteForceSimilarities_k5_run')]
            bruteForceK.sort()  # Ensure order
            MinHashK = [path for path in MinHash if f't{k}_run' in path]
            LSHbaseK = [path for path in LSHbase if f't{k}_run' in path]
        else:
            continue
        
        # Process each run index
        for run_idx in range(1, len(MinHashK)+1):
            # Find files with current run index
            bf_run = [f for f in bruteForceK if f'_run{run_idx}.csv' in f]
            mh_run = [f for f in MinHashK if f'_run{run_idx}.csv' in f]
            lsh_run = [f for f in LSHbaseK if f'_run{run_idx}.csv' in f]
            
            if not bf_run or not mh_run or not lsh_run:
                continue  # Skip missing runs
            
            bruteForceFile = bf_run[0]
            MinHashFile = mh_run[0]
            LSHbaseFile = lsh_run[0]
            
            resultMinHash = get_precision(bruteForceFile, MinHashFile)
            resultLSHbase = get_precision(bruteForceFile, LSHbaseFile)
            
            PrecisionMinHash.append(resultMinHash)
            PrecisionLSHbase.append(resultLSHbase)
    
    # Average results across runs
    avg_MinHash = sum(PrecisionMinHash)/len(PrecisionMinHash) if PrecisionMinHash else 0
    avg_LSHbase = sum(PrecisionLSHbase)/len(PrecisionLSHbase) if PrecisionLSHbase else 0
    
    return [avg_MinHash, avg_LSHbase]


def main():
    parser = argparse.ArgumentParser(
        description='Document Similarity Methods Evaluation')
    parser.add_argument('--mode',
                        choices=['real', 'virtual'],
                        required=True,
                        help='Dataset mode: real or virtual')
    parser.add_argument('--num_docs',
                        type=int,
                        default=300,
                        help='Number of documents to generate')
    parser.add_argument('--prepare_datasets',
                        action='store_true',
                        help='Prepare datasets before running experiments')
    parser.add_argument('--experiment_type',
                        choices=['vary_k', 'vary_t', 'vary_b', 'vary_thr', 'all', 'analyze_only'],
                        default='all',
                        help='Type of experiment to run')
    # Base parameter values
    parser.add_argument('--base_k',
                        type=int,
                        default=5,
                        help='Base value for k (shingle size)')
    parser.add_argument('--base_t',
                        type=int,
                        default=500,
                        help='Base value for t (number of hash functions)')
    parser.add_argument('--base_b',
                        type=int,
                        default=50,
                        help='Base value for b as percentage of t')
    parser.add_argument('--base_thr',
                        type=float,
                        default=0.5,
                        help='Base value for threshold')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='Generate visualizations after experiments')
    parser.add_argument('--num_runs', 
                        type=int,
                        default=5,
                        help='Number of runs for each experiment')

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
    output_dir = os.path.join('results', args.mode)
    os.makedirs(output_dir, exist_ok=True)
    values_to_try = list()
    # Skip running experiments if analyze_only is specified
    if args.experiment_type != 'analyze_only':
        # Run experiments based on the specified type
        if args.experiment_type == 'vary_k' or args.experiment_type == 'all':
            logging.info("Running experiment varying k...")
            csvs = run_parameter_experiment(bin, dataset_dir, output_dir, 'k', 
                                    args.base_k, args.base_t, args.base_b, args.base_thr, args.num_runs)
            values_to_try = list(range(1, 15))
            # print(values_to_try)
            output = precisions_files_var(csvs,'k',values_to_try,args.mode)
            MinHasPrecisions = output[0]
            LSHPrecision = output[1]
            
            
            print(MinHasPrecisions)
            print(LSHPrecision)
            
        if args.experiment_type == 'vary_t' or args.experiment_type == 'all':
            logging.info("Running experiment varying t...")
            csvs = run_parameter_experiment(bin, dataset_dir, output_dir, 't', 
                                    args.base_k, args.base_t, args.base_b, args.base_thr, args.num_runs)
            
            values_to_try = list(range(100, 1001, 100))
            # print(values_to_try)
            output = precisions_files_var(csvs,'t',values_to_try,args.mode)
            MinHasPrecisions = output[0]
            LSHPrecision = output[1]
            print(MinHasPrecisions)
            print(LSHPrecision)
            
        if args.experiment_type == 'vary_b' or args.experiment_type == 'all':
            logging.info("Running experiment varying b...")
            csvs = run_parameter_experiment(bin, dataset_dir, output_dir, 'b', 
                                    args.base_k, args.base_t, args.base_b, args.base_thr, args.num_runs)

            
            
            
        if args.experiment_type == 'vary_thr' or args.experiment_type == 'all':
            logging.info("Running experiment varying threshold...")
            run_parameter_experiment(bin, dataset_dir, output_dir, 'thr', 
                                    args.base_k, args.base_t, args.base_b, args.base_thr, args.num_runs)

        logging.info("Experiments completed successfully.")
    
    # Generate visualizations if requested or if analyze_only
    if args.visualize or args.experiment_type == 'analyze_only':
        logging.info("Generating visualizations...")
        
        # Determine which experiment types to analyze
        if args.experiment_type == 'all' or args.experiment_type == 'analyze_only':
            experiment_types = ['vary_k', 'vary_t', 'vary_b', 'vary_thr']
        else:
            experiment_types = [args.experiment_type]
        
        analyze_and_visualize_results(args.mode, experiment_types)
        logging.info("Visualization completed.")

if __name__ == "__main__":
    main()
