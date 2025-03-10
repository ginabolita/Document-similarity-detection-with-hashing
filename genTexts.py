import random
import argparse
import string
from typing import Set, Tuple, List

def generate_random_text(word_count: int, min_word_length: int = 3, max_word_length: int = 10) -> str:
    """Generate random text with the specified number of words."""
    words = []
    for _ in range(word_count):
        word_length = random.randint(min_word_length, max_word_length)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
        words.append(word)
    return ' '.join(words)

def get_word_shingles(text: str, k: int) -> Set[str]:
    """Extract word-level k-shingles from text."""
    words = text.split()
    shingles = set()
    if len(words) >= k:
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
    return shingles

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def generate_texts_with_target_similarity(target_similarity: float, k: int, tolerance: float = 0.01) -> Tuple[str, str, float]:
    """
    Generate two texts with approximately the target Jaccard similarity using word-level shingles.
    
    Args:
        target_similarity: The desired Jaccard similarity (0.0 to 1.0)
        k: Size of word shingles
        tolerance: Acceptable deviation from target similarity
        
    Returns:
        Tuple of (text1, text2, actual_similarity)
    """
    # Generate a base text
    base_words = 500  # Generate more words to have enough shingles
    text1 = generate_random_text(base_words)
    words1 = text1.split()
    shingles1 = get_word_shingles(text1, k)
    
    # Edge cases
    if target_similarity == 0:
        text2 = generate_random_text(base_words)
        shingles2 = get_word_shingles(text2, k)
        actual_similarity = calculate_jaccard_similarity(shingles1, shingles2)
        return text1, text2, actual_similarity
    
    if target_similarity == 1:
        return text1, text1, 1.0
    
    # For other similarities, we'll blend parts of text1 with new text
    best_text2 = ""
    best_similarity = 0
    best_difference = float('inf')
    
    for attempt in range(1000):
        # Calculate number of shared shingles needed
        # If we have S1 shingles in text1, S2 shingles in text2, and S_shared shared shingles,
        # then similarity = S_shared / (S1 + S2 - S_shared)
        # Assuming S1 ≈ S2 (similar text lengths), we can solve for S_shared:
        # S_shared = target * (2*S1 - S_shared)
        # S_shared * (1 + target) = 2 * target * S1
        # S_shared = (2 * target * S1) / (1 + target)
        
        total_shingles = len(shingles1)
        shared_shingles_needed = int((2 * target_similarity * total_shingles) / (1 + target_similarity))
        shared_shingles_needed = min(shared_shingles_needed, total_shingles)
        
        # Select which shingles to share
        shingles1_list = list(shingles1)
        if shared_shingles_needed > 0:
            shared_shingles = set(random.sample(shingles1_list, shared_shingles_needed))
        else:
            shared_shingles = set()
            
        # Create a new text that shares exactly these shingles
        if shared_shingles:
            # Extract words from shared shingles
            shared_words = set()
            for shingle in shared_shingles:
                for word in shingle.split():
                    shared_words.add(word)
                    
            # Create a word mapping from text1
            word_positions = {}
            for i, word in enumerate(words1):
                if word not in word_positions:
                    word_positions[word] = []
                word_positions[word].append(i)
            
            # Determine which parts of text1 to keep
            # We'll keep all words that are part of a shared shingle
            keep_positions = set()
            for i in range(len(words1) - k + 1):
                shingle = ' '.join(words1[i:i+k])
                if shingle in shared_shingles:
                    for j in range(i, i+k):
                        keep_positions.add(j)
            
            # Create a new text2 that keeps these positions but randomly fills others
            words2 = [None] * len(words1)  # Initialize with None placeholders
            
            # First, keep the words at positions we want to preserve
            for pos in keep_positions:
                words2[pos] = words1[pos]
            
            # Fill in the rest with random words
            for i in range(len(words2)):
                if words2[i] is None:
                    word_length = random.randint(3, 10)
                    words2[i] = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            
            text2 = ' '.join(words2)
        else:
            # For very low similarities, just create a completely different text
            text2 = generate_random_text(base_words)
            
        # Calculate actual similarity
        shingles2 = get_word_shingles(text2, k)
        actual_similarity = calculate_jaccard_similarity(shingles1, shingles2)
        
        # Check if we're within tolerance
        difference = abs(actual_similarity - target_similarity)
        if difference <= tolerance:
            return text1, text2, actual_similarity
            
        # Track the best attempt
        if difference < best_difference:
            best_difference = difference
            best_text2 = text2
            best_similarity = actual_similarity
    
    print(f"Warning: Could not achieve exact target similarity. Best: {best_similarity:.4f}")
    return text1, best_text2, best_similarity

def main():
    parser = argparse.ArgumentParser(description='Generate two texts with a target Jaccard similarity')
    parser.add_argument('--target', type=float, default=0.5, help='Target Jaccard similarity (0.0-1.0)')
    parser.add_argument('--shingle-size', type=int, default=3, help='Size of word shingles (k)')
    parser.add_argument('--output1', type=str, default='text1.txt', help='Output file for first text')
    parser.add_argument('--output2', type=str, default='text2.txt', help='Output file for second text')
    parser.add_argument('--tolerance', type=float, default=0.02, help='Acceptable deviation from target similarity')
    parser.add_argument('--verify', action='store_true', help='Print verification information')
    
    args = parser.parse_args()
    
    if args.target < 0 or args.target > 1:
        print("Error: Target similarity must be between 0.0 and 1.0")
        return
    
    text1, text2, actual_similarity = generate_texts_with_target_similarity(
        args.target, args.shingle_size, args.tolerance
    )
    
    # Save the texts to files
    with open(args.output1, 'w') as f1:
        f1.write(text1)
    
    with open(args.output2, 'w') as f2:
        f2.write(text2)
    
    print(f"\nGenerated two texts with Jaccard similarity: {actual_similarity:.4f}")
    print(f"Target similarity was: {args.target:.4f}")
    
    if args.verify:
        # Print verification information
        shingles1 = get_word_shingles(text1, args.shingle_size)
        shingles2 = get_word_shingles(text2, args.shingle_size)
        intersection = shingles1.intersection(shingles2)
        
        print(f"\nText 1 words: {len(text1.split())}")
        print(f"Text 2 words: {len(text2.split())}")
        print(f"Text 1 unique {args.shingle_size}-shingles: {len(shingles1)}")
        print(f"Text 2 unique {args.shingle_size}-shingles: {len(shingles2)}")
        print(f"Shared shingles: {len(intersection)}")
        print(f"Union of shingles: {len(shingles1.union(shingles2))}")
        
        # Print sample of shared shingles
        if intersection:
            print("\nSample of shared shingles:")
            for shingle in list(intersection)[:5]:
                print(f"  - '{shingle}'")
    
    print(f"\nText 1 saved to: {args.output1}")
    print(f"Text 2 saved to: {args.output2}")
    print(f"Word shingle size used: {args.shingle_size}")
    
    # Print instruction to verify with C++ program
    print("\nNotes for using with your C++ program:")
    print("1. Your C++ program uses character-level shingles by default")
    print("2. To use word-level shingles, you would need to modify your C++ code")
    print("3. Different shingle types will produce different similarity values")

if __name__ == "__main__":
    main()