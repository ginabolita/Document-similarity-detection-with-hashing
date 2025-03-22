import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# parse metadata that contains the name of the .csv
def parse_fileMetadata(filename):
    filename = filename.replace('.csv', '')  # remove the .csv extension

    parts = filename.split('_')

    # The last 4 parts should be parameters, but some might be "NA"
    # Get all parts except potential parameters (which should be the last 4)
    if len(parts) >= 4:
        algorithm = '_'.join(parts[:-4])
        param_parts = parts[-4:]
    else:
        # If we have fewer than 4 parts, assume it's just the algorithm name
        algorithm = filename
        param_parts = []

    # Initialize parameters with None to indicate they're not present
    k = None
    t = None
    b = None
    thr = None

    # Parse parameters from the parts
    for part in param_parts:
        if part.startswith('k'):
            k = float(part.replace('k', '')) if part != 'kNA' else None
        elif part.startswith('t') and not part.startswith('th'):
            t = float(part.replace('t', '')) if part != 'tNA' else None
        elif part.startswith('b'):
            b = float(part.replace('b', '')) if part != 'bNA' else None
        elif part.startswith('th'):
            thr = float(part.replace('th', '')) if part != 'thNA' else None

    return algorithm, k, t, b, thr


# analysis for each .csv
def single_file_analysis(filepath, output_dir, algorithm, k, t, b, thr):
    # Try different ways to read the CSV file
    try:
        # First attempt with default settings
        data = pd.read_csv(filepath)

        # Check if 'Sim%' column exists, otherwise try to find it with different naming
        if 'Sim%' not in data.columns:
            print(f"  Warning: 'Sim%' column not found in {filepath}")
            sim_column = None

            # Check for possible alternative column names
            possible_names = [
                'Similarity', 'Similarity%', 'SimPercent', 'Sim', 'sim%',
                'similarity'
            ]
            for name in possible_names:
                if name in data.columns:
                    print(f"  Using '{name}' column instead of 'Sim%'")
                    sim_column = name
                    break

            # If no similarity column is found, try to find a numeric column
            if sim_column is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sim_column = numeric_cols[0]
                    print(
                        f"  Using numeric column '{sim_column}' instead of 'Sim%'"
                    )

            # If still no column found, create a dummy one for basic analysis
            if sim_column is None:
                print(
                    f"  No suitable column found. Creating a dummy 'Sim%' column with zeros"
                )
                data['Sim%'] = 0
            else:
                # Rename the found column to 'Sim%' for consistent processing
                data = data.rename(columns={sim_column: 'Sim%'})
    except pd.errors.ParserError:
        # If standard parsing fails, try with different separators
        print(f"  CSV parsing error. Trying alternative delimiter...")
        try:
            # Try with tab delimiter
            data = pd.read_csv(filepath, sep='\t')
            if 'Sim%' not in data.columns and len(data.columns) >= 3:
                # If we have multiple columns but no 'Sim%', rename the columns
                data.columns = ['Doc1', 'Doc2', 'Sim%'] + list(
                    data.columns[3:])
                print(f"  Renamed columns to ['Doc1', 'Doc2', 'Sim%', ...]")
        except Exception:
            # Try with semicolon delimiter (common in European CSV files)
            try:
                data = pd.read_csv(filepath, sep=';')
                if 'Sim%' not in data.columns and len(data.columns) >= 3:
                    data.columns = ['Doc1', 'Doc2', 'Sim%'] + list(
                        data.columns[3:])
                    print(
                        f"  Renamed columns to ['Doc1', 'Doc2', 'Sim%', ...]")
            except Exception as e:
                # Create a minimal DataFrame with dummy data if all parsing fails
                print(
                    f"  All parsing attempts failed. Creating a dummy DataFrame for minimal analysis"
                )
                data = pd.DataFrame({
                    'Doc1': ['doc1', 'doc2'],
                    'Doc2': ['doc2', 'doc1'],
                    'Sim%': [0, 0]
                })

    # Ensure Doc1 and Doc2 columns exist
    if 'Doc1' not in data.columns or 'Doc2' not in data.columns:
        print(
            f"  Doc1 or Doc2 columns missing. Creating them from row indices.")
        if 'Doc1' not in data.columns:
            data['Doc1'] = [f"Doc{i}" for i in range(len(data))]
        if 'Doc2' not in data.columns:
            data['Doc2'] = [f"Doc{i+1}" for i in range(len(data))]

    # Create the directory where we save the outputs
    # Format the parameters, using "NA" for None values
    k_str = f"k{k}" if k is not None else "kNA"
    t_str = f"t{t}" if t is not None else "tNA"
    b_str = f"b{b}" if b is not None else "bNA"
    thr_str = f"th{thr}" if thr is not None else "thNA"

    subdirectory = f"{algorithm}_{k_str}_{t_str}_{b_str}_{thr_str}"
    output_path = os.path.join(output_dir, subdirectory)
    os.makedirs(output_path, exist_ok=True)

    # Handle missing values in the data
    data = data.fillna(0)  # Replace NA with 0 for numerical operations

    # Convert Sim% to numeric if it isn't already
    data['Sim%'] = pd.to_numeric(data['Sim%'], errors='coerce').fillna(0)

    # Create title with parameters (showing NA for missing ones)
    title_params = []
    if k is not None:
        title_params.append(f"k={k}")
    if t is not None:
        title_params.append(f"t={t}")
    if b is not None:
        title_params.append(f"b={b}")
    if thr is not None:
        title_params.append(f"thr={thr}")

    params_str = ", ".join(title_params) if title_params else "No parameters"

    # Plot 1: Histogram of Similarity Percentages
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Sim%'], bins=20, kde=True)
        plt.title(
            f'Distribution of Similarity Percentages\n{algorithm}, {params_str}'
        )
        plt.xlabel('Similarity Percentage')
        plt.ylabel('Frequency')
        plt.savefig(f"{output_path}/histogram.png")
        plt.close()
    except Exception as e:
        print(f"  Error creating histogram: {e}")

    # Plot 2: Boxplot of Similarity Percentages
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data['Sim%'])
        plt.title(
            f'Boxplot of Similarity Percentages\n{algorithm}, {params_str}')
        plt.xlabel('Similarity Percentage')
        plt.savefig(f"{output_path}/boxplot.png")
        plt.close()
    except Exception as e:
        print(f"  Error creating boxplot: {e}")

    # Plot 3: Heatmap of Similarity Percentages
    # Check if there are enough unique Doc1 and Doc2 entries to create a meaningful heatmap
    if len(data['Doc1'].unique()) > 1 and len(data['Doc2'].unique()) > 1:
        try:
            pivot_table = data.pivot(index='Doc1',
                                     columns='Doc2',
                                     values='Sim%')
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='viridis')
            plt.title(
                f'Heatmap of Similarity Percentages\n{algorithm}, {params_str}'
            )
            plt.xlabel('Doc2')
            plt.ylabel('Doc1')
            plt.savefig(f"{output_path}/heatmap.png")
            plt.close()
        except Exception as e:
            print(f"  Error creating heatmap: {e}")
    else:
        print(f"  Not enough unique documents to create a heatmap")

    # Plot 4: Scatter plot of Doc1 vs Doc2 with Sim% as color
    try:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data['Doc1'],
                              data['Doc2'],
                              c=data['Sim%'],
                              cmap='viridis',
                              alpha=0.6)
        plt.colorbar(scatter, label='Similarity Percentage')
        plt.title(f'Scatter Plot of Doc1 vs Doc2\n{algorithm}, {params_str}')
        plt.xlabel('Doc1')
        plt.ylabel('Doc2')
        plt.savefig(f"{output_path}/scatter_plot.png")
        plt.close()
    except Exception as e:
        print(f"  Error creating scatter plot: {e}")

    # Plot 5: Bar plot of average similarity percentage per document
    try:
        avg_sim_doc1 = data.groupby('Doc1')['Sim%'].mean().reset_index()
        avg_sim_doc2 = data.groupby('Doc2')['Sim%'].mean().reset_index()

        plt.figure(figsize=(12, 6))
        plt.bar(avg_sim_doc1['Doc1'],
                avg_sim_doc1['Sim%'],
                alpha=0.6,
                label='Doc1')
        plt.bar(avg_sim_doc2['Doc2'],
                avg_sim_doc2['Sim%'],
                alpha=0.6,
                label='Doc2')
        plt.title(
            f'Average Similarity Percentage per Document\n{algorithm}, {params_str}'
        )
        plt.xlabel('Document')
        plt.ylabel('Average Similarity Percentage')
        plt.legend()
        plt.savefig(f"{output_path}/bar_plot.png")
        plt.close()
    except Exception as e:
        print(f"  Error creating bar plot: {e}")

    # Save summary statistics to a text file
    try:
        with open(f"{output_path}/summary.txt", "w") as f:
            f.write(f"Algorithm: {algorithm}\n")
            if k is not None:
                f.write(f"k: {k}\n")
            else:
                f.write("k: NA\n")

            if t is not None:
                f.write(f"t: {t}\n")
            else:
                f.write("t: NA\n")

            if b is not None:
                f.write(f"b: {b}\n")
            else:
                f.write("b: NA\n")

            if thr is not None:
                f.write(f"thr: {thr}\n")
            else:
                f.write("thr: NA\n")

            f.write("\nSummary Statistics:\n")
            f.write(data['Sim%'].describe().to_string())

            # Save the data preview
            f.write("\n\nData Preview (first 5 rows):\n")
            f.write(data.head(5).to_string())

            # Save column information
            f.write("\n\nColumns in the file:\n")
            for col in data.columns:
                f.write(f"- {col}\n")
    except Exception as e:
        print(f"  Error writing summary file: {e}")

    # Also save the processed data for reference
    try:
        data.to_csv(f"{output_path}/processed_data.csv", index=False)
    except Exception as e:
        print(f"  Error saving processed data: {e}")


# process all .csv files in the directory
def analyze_directory(source_dir, output_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(source_dir, filename)
            try:
                algorithm, k, t, b, thr = parse_fileMetadata(filename)
                print(f"Analyzing {filename}...")
                print(
                    f"  Parsed parameters: algorithm={algorithm}, k={k}, t={t}, b={b}, thr={thr}"
                )
                single_file_analysis(filepath, output_dir, algorithm, k, t, b,
                                     thr)
                print(f"  Analysis complete for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CSV files containing document similarity data.",
        usage="python3 analysis.py <source_dir> <output_dir>")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Path to the directory containing the CSV files to analyze.")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory where the analysis results will be saved.")
    args = parser.parse_args()

    # check if the directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # analyze each csv in the directory
    analyze_directory(args.source_dir, args.output_dir)
    print(f"Analysis complete. Results saved in {args.output_dir}.")


if __name__ == "__main__":
    main()
