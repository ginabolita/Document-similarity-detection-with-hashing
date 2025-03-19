import itertools
import os
import subprocess
import csv
import re

def extract_doc_number(filename):
    match = re.search(r'docExp1_(\d+)', filename)
    return match.group(1) if match else filename

def run_jaccard_brute_force(directory, executable, constant_arg, output_file):
    # Get all files in the directory
    files = sorted([f for f in os.listdir(directory) if f.startswith("docExp")])
    
    # Generate all unique pairs of files
    file_pairs = list(itertools.combinations(files, 2))
    
    results = []
    for file1, file2 in file_pairs:
        command = f"./{executable} {directory}/{file1} {directory}/{file2} {constant_arg}"
        print(f"Running: {command}")
        
        # Execute the command and capture output
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        # Store results with only the extracted 'n' values
        similarity_score = result.stdout.strip()
        results.append([extract_doc_number(file1), extract_doc_number(file2), similarity_score])
    
    # Save results to CSV file
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Doc1", "Doc2", "SimilarityScore"])
        writer.writerows(results)
    
    print(f"Results saved in {output_file}")

if __name__ == "__main__":
    run_jaccard_brute_force("MartinLutherKing", "jaccardBruteForce", 3, "results.csv")
