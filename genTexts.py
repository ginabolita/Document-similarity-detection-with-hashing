import json
import random
import os
import argparse
import re
from pathlib import Path
from typing import List, Set, Dict, Tuple

class DocumentGenerator:
    def __init__(self, stopwords_file: str = "stopwords-en.json"):
        """Initialize the document generator with stopwords file."""
        self.stopwords = self.load_stopwords(stopwords_file)
        self.ensure_directories()
        
    def load_stopwords(self, filename: str) -> Set[str]:
        """Load stopwords from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return set(json.load(file))
        except FileNotFoundError:
            print(f"Warning: Stopwords file {filename} not found. Creating empty set.")
            return set()

    def ensure_directories(self):
        """Ensure output directories exist."""
        Path("datasets/real").mkdir(parents=True, exist_ok=True)
        Path("datasets/virtual").mkdir(parents=True, exist_ok=True)
    
    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word.lower() in self.stopwords
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words, removing stopwords and lowercasing."""
        # Tokenize and filter stopwords
        words = []
        for word in re.findall(r'\b\w+\b', text.lower()):
            if word and not self.is_stopword(word):
                words.append(word)
        return words
    
    def count_unique_words(self, words: List[str]) -> int:
        """Count unique words in a list."""
        return len(set(words))
    
    def generate_random_permutations(self, words: List[str], num_docs: int) -> List[str]:
        """Generate random permutations of words."""
        permutations = []
        
        for _ in range(num_docs):
            # Create a copy for permutation
            temp_words = words.copy()
            random.shuffle(temp_words)
            
            # Combine shuffled words into a document
            shuffled_text = []
            for i, word in enumerate(temp_words):
                shuffled_text.append(word)
                
                # Add space and occasionally add periods to create sentences
                if i < len(temp_words) - 1:
                    if random.randint(0, 9) == 0:  # ~10% chance for a period
                        shuffled_text.append(".")
                    shuffled_text.append(" ")
            
            # Ensure document ends with a period
            if shuffled_text and shuffled_text[-1] != ".":
                shuffled_text.append(".")
                
            permutations.append("".join(shuffled_text))
        
        return permutations

    def generate_shingles(self, words: List[str], k: int) -> Set[str]:
        """Generate k-word shingles from a list of words."""
        shingles = set()
        
        if len(words) >= k:
            for i in range(len(words) - k + 1):
                shingle = " ".join(words[i:i+k])
                shingles.add(shingle)
        
        print(f"Total {k}-shingles generated: {len(shingles)}")
        return shingles
    
    def select_random_shingles(self, shingles: Set[str], quantity: int) -> List[str]:
        """Select a random subset of shingles."""
        shingles_list = list(shingles)
        if quantity >= len(shingles_list):
            return shingles_list
        
        return random.sample(shingles_list, quantity)
    
    def calculate_expected_similarity(self, n_i: int, n_j: int, n: int) -> float:
        """Calculate expected similarity based on formula in the PDF."""
        p_i = n_i / n
        p_j = n_j / n
        return (p_i * p_j) / (p_i + p_j - p_i * p_j)
    
    def generate_real_documents(self, base_text: str, num_docs: int) -> None:
        """Generate 'real' documents (Experiment 1)."""
        words = self.tokenize_words(base_text)
        unique_word_count = self.count_unique_words(words)
        
        if unique_word_count < 50:
            print(f"Error: Base text must contain at least 50 different words. Current count: {unique_word_count}")
            return
        
        print(f"Base text contains {len(words)} total words and {unique_word_count} unique words.")
        
        # Generate permutations at word level
        permutations = self.generate_random_permutations(words, num_docs)
        
        # Write to files
        for i, permutation in enumerate(permutations):
            filename = f"datasets/real/docExp1_{i+1}.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(permutation)
            print(f"Generated file: {filename}")
    
    def generate_virtual_documents(self, base_text: str, k: int, num_docs: int) -> None:
        """Generate 'virtual' documents (Experiment 2)."""
        words = self.tokenize_words(base_text)
        unique_word_count = self.count_unique_words(words)
        
        if unique_word_count < 100:
            print(f"Error: Base text for experiment 2 must contain at least 100 different words. Current count: {unique_word_count}")
            return
        
        print(f"Base text contains {unique_word_count} unique words.")
        
        # Generate shingles
        shingles = self.generate_shingles(words, k)
        total_shingles = len(shingles)
        
        if total_shingles == 0:
            print("Error: No shingles could be generated. Text might be too short for the given k value.")
            return
        
        # Define shingle count ranges
        min_shingles = max(10, total_shingles // 10)  # At least 10 or 10% of total
        max_shingles = min(total_shingles * 8 // 10, total_shingles - 1)  # At most 80% of total
        
        shingle_counts = []
        
        # Generate documents
        for i in range(num_docs):
            filename = f"datasets/virtual/docExp2_{i+1}.txt"
            
            # Random quantity between min_shingles and max_shingles
            rand_quantity = random.randint(min_shingles, max_shingles)
            selected_shingles = self.select_random_shingles(shingles, rand_quantity)
            shingle_counts.append(rand_quantity)
            
            with open(filename, 'w', encoding='utf-8') as file:
                for shingle in selected_shingles:
                    file.write(f"{shingle}\n")
            
            print(f"Generated file: {filename} with {rand_quantity} shingles")
        
        # Generate similarity matrix report
        with open("datasets/similarity_matrix.txt", 'w', encoding='utf-8') as sim_matrix:
            sim_matrix.write("Expected Similarity Matrix between documents:\n")
            sim_matrix.write(f"Total k-shingles in base set: {total_shingles}\n\n")
            
            # Create table header
            sim_matrix.write("Doc\t")
            for i in range(num_docs):
                sim_matrix.write(f"Doc{i+1}\t")
            sim_matrix.write("\n")
            
            # Create similarity matrix
            for i in range(num_docs):
                sim_matrix.write(f"Doc{i+1}\t")
                for j in range(num_docs):
                    if i == j:
                        sim_matrix.write("1.000\t")  # Self-similarity is 1
                    else:
                        similarity = self.calculate_expected_similarity(
                            shingle_counts[i], shingle_counts[j], total_shingles)
                        sim_matrix.write(f"{similarity:.3f}\t")
                sim_matrix.write("\n")
            
            print("Generated similarity matrix")

def load_json_text(json_file: str, experiment: int) -> str:
    """Load base text from a JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data[f"experimento_{experiment}"]["basicText"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading base text: {e}")
        return ""

def create_sample_json():
    """Create a sample basicText.json file if it doesn't exist."""
    if os.path.exists("basicText.json"):
        return
    
    # Sample text with well over 100 unique words (for both experiments)
    sample_text = """
    Machine learning is a branch of artificial intelligence that focuses on developing systems
    that can learn from and make decisions based on data. These algorithms can improve their performance
    over time as they are exposed to more information. The field has grown significantly in the past decade,
    with applications ranging from image recognition and natural language processing to autonomous vehicles
    and medical diagnosis. Deep learning, a subset of machine learning, uses neural networks with many layers
    to process complex patterns in large datasets. Researchers continue to explore new architectures and techniques
    to enhance model accuracy and efficiency. As these technologies become more accessible, they are transforming
    industries and creating new opportunities for innovation. The ethical implications of these powerful tools
    are also being widely discussed, particularly regarding privacy, bias, and the potential impact on employment.
    Striking a balance between technological advancement and responsible deployment remains a key challenge for
    the field moving forward. Open source communities have been instrumental in democratizing access to cutting-edge
    machine learning resources. Frameworks such as TensorFlow, PyTorch, and scikit-learn have made it easier for
    developers to implement sophisticated algorithms without extensive mathematical expertise. Cloud computing has
    further reduced barriers to entry by providing scalable infrastructure for training and deploying models.
    As machine learning continues to evolve, interdisciplinary collaboration between computer scientists, statisticians,
    domain experts, and ethicists will be essential for addressing complex problems and ensuring beneficial outcomes for society.
    """
    
    json_data = {
        "experimento_1": {"basicText": sample_text},
        "experimento_2": {"basicText": sample_text}
    }
    
    with open("basicText.json", 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=2)
    
    print("Created sample basicText.json file with sufficient unique words")

def create_sample_stopwords():
    """Create a sample stopwords file if it doesn't exist."""
    if os.path.exists("stopwords-en.json"):
        return
    
    # Common English stopwords
    stopwords = [
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
        "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
        "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my",
        "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
        "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
        "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
        "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    ]
    
    with open("stopwords-en.json", 'w', encoding='utf-8') as file:
        json.dump(stopwords, file, indent=2)
    
    print("Created sample stopwords-en.json file")

def main():
    parser = argparse.ArgumentParser(description="Document generator for Jaccard similarity experiments")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], default=3,
                        help="Experiment to run: 1=real docs, 2=virtual docs, 3=both (default)")
    parser.add_argument("--k", type=int, default=3, help="Shingle size for experiment 2 (default: 3)")
    parser.add_argument("--D", type=int, default=20, help="Number of documents to generate (default: 20)")
    parser.add_argument("--json_file", type=str, default="basicText.json", help="JSON file with base texts")
    parser.add_argument("--stopwords", type=str, default="stopwords-en.json", help="JSON file with stopwords")
    
    args = parser.parse_args()
    
    # Create sample files if needed
    create_sample_stopwords()
    create_sample_json()
    
    # Initialize document generator
    generator = DocumentGenerator(stopwords_file=args.stopwords)
    
    # Run experiments
    if args.experiment in [1, 3]:
        print("\n== Running Experiment 1: 'Real' Documents ==")
        base_text = load_json_text(args.json_file, 1)
        if base_text:
            generator.generate_real_documents(base_text, args.D)
    
    if args.experiment in [2, 3]:
        print("\n== Running Experiment 2: 'Virtual' Documents ==")
        base_text = load_json_text(args.json_file, 2)
        if base_text:
            generator.generate_virtual_documents(base_text, args.k, args.D)

if __name__ == "__main__":
    main()