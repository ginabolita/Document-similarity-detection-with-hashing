import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

sns.set(style="whitegrid")


# parse metadata that contains the name of the .csv
def parse_fileMetadata(filename):
    filename = filename.replace('.csv', '')
    parts = filename.split('_')

    if len(parts) >= 4:
        algorithm = '_'.join(parts[:-4])
        param_parts = parts[-4:]
    else:
        algorithm = filename
        param_parts = []

    k, t, b, thr = None, None, None, None

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


def single_file_analysis(filepath, output_dir, algorithm, k, t, b, thr):
    try:
        data = pd.read_csv(filepath)
        if 'Sim%' not in data.columns:
            possible_names = ['Similarity', 'Similarity%', 'SimPercent', 'Sim', 'sim%', 'similarity']
            for name in possible_names:
                if name in data.columns:
                    data = data.rename(columns={name: 'Sim%'})
                    break
        data['Sim%'] = pd.to_numeric(data['Sim%'], errors='coerce').fillna(0)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    output_path = os.path.join(output_dir, algorithm)
    os.makedirs(output_path, exist_ok=True)

    try:
        # Histograma
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Sim%'], bins=20, kde=True, color="cornflowerblue")
        plt.title(f"Similarity Distribution - {algorithm}")
        plt.xlabel("Similarity (%)")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_path}/histogram.png")
        plt.close()

        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data['Sim%'], color="lightgreen")
        plt.title(f"Boxplot of Similarity - {algorithm}")
        plt.xlabel("Similarity (%)")
        plt.savefig(f"{output_path}/boxplot.png")
        plt.close()

        # Heatmap (original y de diferencias absolutas)
        if 'Doc1' in data.columns and 'Doc2' in data.columns:
            # Heatmap original
            pivot_table = data.pivot(index='Doc1', columns='Doc2', values='Sim%')
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=False, cmap='viridis')
            plt.title(f"Heatmap of Similarity - {algorithm}")
            plt.savefig(f"{output_path}/heatmap.png")
            plt.close()

            # Heatmap de diferencias absolutas (solo si no es bruteForce)
            if algorithm != "bruteForce":
                # Construir el nombre del archivo bruteForce
                brute_force_filename = f"bruteForceSimil       arities_k{k}.csv"
                brute_force_filepath = os.path.join("results/virtual/bruteForce/", brute_force_filename)

                if os.path.exists(brute_force_filepath):
                    brute_force_data = pd.read_csv(brute_force_filepath)
                    if 'Sim%' not in brute_force_data.columns:
                        for name in possible_names:
                            if name in brute_force_data.columns:
                                brute_force_data = brute_force_data.rename(columns={name: 'Sim%'})
                                break
                    brute_force_data['Sim%'] = pd.to_numeric(brute_force_data['Sim%'], errors='coerce').fillna(0)

                    # Crear un DataFrame con las diferencias absolutas
                    diff_data = data.copy()
                    diff_data['Sim%'] = abs(data['Sim%'] - brute_force_data['Sim%'])

                    # Generar el heatmap con las diferencias absolutas
                    pivot_table_diff = diff_data.pivot(index='Doc1', columns='Doc2', values='Sim%')
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(pivot_table_diff, annot=False, cmap='viridis')
                    plt.title(f"Heatmap of Absolute Similarity Difference - {algorithm}")
                    plt.savefig(f"{output_path}/heatmap_diff.png")
                    plt.close()
                else:
                    print(f"BruteForce file not found at {brute_force_filepath}")

    except Exception as e:
        print(f"Error creating plot: {e}")


def analyze_directory(source_dir, output_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(source_dir, filename)
            algorithm, k, t, b, thr = parse_fileMetadata(filename)
            print(f"Analyzing {filename}...")
            single_file_analysis(filepath, output_dir, algorithm, k, t, b, thr)
            print(f"  Analysis complete for {filename}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CSV files with document similarity data.")
    parser.add_argument("source_dir", type=str, help="Directory containing the CSV files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the analysis results.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_directory(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
