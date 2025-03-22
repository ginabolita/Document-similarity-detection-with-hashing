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
    if t is not None:
        param_parts.append(f"t{t}")
    if b is not None:
        param_parts.append(f"b{b}")
    if thr is not None:
        param_parts.append(f"threshold{thr}")
    
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
        logging.info("entra en if")
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

    # Add parameters if provided for other executables
    if k is not None:
        cmd.append(str(k))
    if t is not None:
        cmd.append(str(t))
    if b is not None:
        cmd.append(str(b))
    if thr is not None:
        cmd.append(str(thr))

    try:
        start_time = time.time()
        # Execute the command
        print(cmd)
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        # Log the run
        logging.info(
            f"Successfully ran corpus mode {executable_path} on {dataset_path}"
        )

        # Get output CSV files
        similarity_csv = os.path.join(output_dir, f"{algo_type}/{algo_type}Similarities_{'_'.join(param_parts)}.csv")
        times_csv = os.path.join(output_dir, f"{algo_type}/{algo_type}Times_{'_'.join(param_parts)}.csv")

        print(f"Output CSV files: {similarity_csv}, {times_csv}")

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
        print(result['similarity_csv'])
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
            else:
                logging.error(f"No similar pairs found for {result['method']}")
        else:
            logging.error(f"Similarity CSV not found for {result['method']}")
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
                        elif "time" in task.lower() and time_value is not None:
                            total_runtime = float(time_value)
                    else:
                        logging.error(f"Invalid task name in times CSV: {task}")
            else:
                logging.error(f"No timing information found for {result['method']}")
        else:
            logging.error(f"Times CSV not found for {result['method']}")
    except Exception as e:
        logging.error(f"Error parsing times CSV: {e}")

    return {
        'dataset': result['dataset'],
        'similar_pairs': similar_pairs,
        'index_build_time': index_build_time,
        'query_time': query_time,
        'total_runtime': total_runtime,
        'status': result['status'],
        'method': result['method'],
        'k': result['k'],
        't': result['t'],
        'b': result['b'],
        'thr': result['thr']
    }


def run_parameter_experiment(bin, dataset_dir, output_dir, param_to_vary, 
                            base_k=5, base_t=500, base_b=50, base_thr=0.3):
    """Run experiments varying one parameter while fixing others"""
    results = []
    
    # Convert base_b from percentage to actual value based on base_t
    base_b_value = int(base_t * (base_b / 100.0)) if base_b is not None else None
    
    logging.info(f"Running experiment varying {param_to_vary}")
    
    for exec_name, exec_path in bin.items():
        logging.info(f"Running {exec_name} with varying {param_to_vary}")
        
        # Define which parameters to use based on algorithm type
        uses_t = exec_name != 'brute_force'
        uses_b = 'lsh' in exec_name
        uses_thr = exec_name not in ['minhash', 'lsh_basic', 'brute_force']
        
        # Set parameter values for this algorithm
        k_val = base_k
        t_val = base_t if uses_t else None
        b_val = base_b_value if uses_b else None
        thr_val = base_thr if uses_thr else None
        
        # Get parameter values to vary
        if param_to_vary == 'k':
            values_to_try = [3, 5, 7, 9]
        elif param_to_vary == 't' and uses_t:
            values_to_try = [300, 500, 700]
        elif param_to_vary == 'b' and uses_b:
            # Convert percentage to actual values
            values_to_try = [int(base_t * (pct / 100.0)) for pct in [30, 50, 70]]
        elif param_to_vary == 'thr' and uses_thr:
            values_to_try = [0.4, 0.5, 0.6]
        else:
            values_to_try = []  # Skip if parameter doesn't apply to this algorithm
            
        # Run experiments with varying parameter
        for val in values_to_try:
            # Set the parameter to vary
            if param_to_vary == 'k':
                k_val = val
            elif param_to_vary == 't':
                t_val = val
                # Update b since it depends on t
                if uses_b:
                    b_val = int(val * (base_b / 100.0))
            elif param_to_vary == 'b':
                b_val = val
            elif param_to_vary == 'thr':
                thr_val = val
                
            # Run with current parameter values
            result = run_corpus_mode(exec_path, dataset_dir, output_dir, 
                                    k_val, t_val, b_val, thr_val)
            
            # Parse results
            parsed_result = parse_csv_results(result)
            parsed_result['similar_pairs_count'] = len(parsed_result['similar_pairs'])
            parsed_result['varied_param'] = param_to_vary
            parsed_result['varied_value'] = val
            
            results.append(parsed_result)
    
    # Create DataFrame with results
    df = pd.DataFrame(results)
    
    # Save results to CSV
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
                        choices=['vary_k', 'vary_t', 'vary_b', 'vary_thr', 'all'],
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

    # Run experiments based on the specified type
    if args.experiment_type == 'vary_k' or args.experiment_type == 'all':
        logging.info("Running experiment varying k...")
        run_parameter_experiment(bin, dataset_dir, output_dir, 'k', 
                                args.base_k, args.base_t, args.base_b, args.base_thr)
        
    if args.experiment_type == 'vary_t' or args.experiment_type == 'all':
        logging.info("Running experiment varying t...")
        run_parameter_experiment(bin, dataset_dir, output_dir, 't', 
                                args.base_k, args.base_t, args.base_b, args.base_thr)
        
    if args.experiment_type == 'vary_b' or args.experiment_type == 'all':
        logging.info("Running experiment varying b...")
        run_parameter_experiment(bin, dataset_dir, output_dir, 'b', 
                                args.base_k, args.base_t, args.base_b, args.base_thr)
        
    if args.experiment_type == 'vary_thr' or args.experiment_type == 'all':
        logging.info("Running experiment varying threshold...")
        run_parameter_experiment(bin, dataset_dir, output_dir, 'thr', 
                                args.base_k, args.base_t, args.base_b, args.base_thr)

    logging.info("Experiments completed successfully.")



if __name__ == "__main__":
    main()