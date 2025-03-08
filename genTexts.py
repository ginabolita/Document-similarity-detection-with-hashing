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

def get_character_shingles(text: str, k: int) -> Set[str]:
    """Extract character-level k-shingles from text."""
    shingles = set()
    if len(text) >= k:
        for i in range(len(text) - k + 1):
            shingle = text[i:i+k]
            shingles.add(shingle)
    return shingles

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def generate_texts_with_target_similarity(target_similarity: float, k: int, tolerance: float = 0.01) -> Tuple[str, str, float]:
    """
    Generate two texts with approximately the target Jaccard similarity using character-level shingles.
    
    Args:
        target_similarity: The desired Jaccard similarity (0.0 to 1.0)
        k: Size of character shingles
        tolerance: Acceptable deviation from target similarity
        
    Returns:
        Tuple of (text1, text2, actual_similarity)
    """
    # Generate a longer base text to ensure we have enough shingles
    base_words = 200
    text1 = generate_random_text(base_words)
    shingles1 = get_character_shingles(text1, k)
    
    # If target similarity is 0, create completely different text
    if target_similarity == 0:
        text2 = generate_random_text(base_words)
        shingles2 = get_character_shingles(text2, k)
        actual_similarity = calculate_jaccard_similarity(shingles1, shingles2)
        return text1, text2, actual_similarity
    
    # If target similarity is 1, return the same text
    if target_similarity == 1:
        return text1, text1, 1.0
    
    # For any other similarity, we need to create a partially similar text
    shingles1_list = list(shingles1)
    
    # Calculate how many shared shingles we need
    # If we have S1 shingles in text1, and want similarity s
    # and will have S2 shingles in text2,
    # then s = |S1 ∩ S2| / |S1 ∪ S2| = |S1 ∩ S2| / (|S1| + |S2| - |S1 ∩ S2|)
    
    best_text2 = ""
    best_similarity = 0
    best_difference = float('inf')
    
    for attempt in range(100):  # Try multiple approaches
        # We'll determine approximately how many shared shingles we need
        # Let's say we'll have |S1| total shingles for each text, and |S1∩S2| shared shingles
        # Then: s = |S1∩S2| / (2*|S1| - |S1∩S2|)
        # Solving for |S1∩S2|: |S1∩S2| = 2s*|S1| / (1 + s)
        shared_count = int((2 * target_similarity * len(shingles1)) / (1 + target_similarity))
        shared_count = min(shared_count, len(shingles1))
        
        # Select shared shingles
        shared_shingles = set(random.sample(shingles1_list, shared_count))
        
        # Generate a new text2
        # Start with a random text
        text2 = generate_random_text(base_words)
        shingles2 = get_character_shingles(text2, k)
        
        # Calculate current similarity
        current_similarity = calculate_jaccard_similarity(shingles1, shingles2)
        
        # If we need to increase similarity, replace parts of text2 with parts of text1
        if current_similarity < target_similarity:
            # Convert texts to character lists for easier manipulation
            text2_chars = list(text2)
            text1_chars = list(text1)
            
            # We'll replace random sections of text2 with sections from text1
            # Gradually increase section length until we reach desired similarity
            for section_length in range(k, len(text1) // 2, k):
                for i in range(10):  # Try several replacement points
                    if len(text2_chars) <= section_length:
                        continue
                        
                    # Choose a random position in text2 to replace
                    pos2 = random.randint(0, len(text2_chars) - section_length)
                    
                    # Choose a random section from text1
                    pos1 = random.randint(0, len(text1_chars) - section_length)
                    
                    # Replace the section
                    original_section = text2_chars[pos2:pos2+section_length]
                    text2_chars[pos2:pos2+section_length] = text1_chars[pos1:pos1+section_length]
                    
                    # Calculate new similarity
                    new_text2 = ''.join(text2_chars)
                    new_shingles2 = get_character_shingles(new_text2, k)
                    new_similarity = calculate_jaccard_similarity(shingles1, new_shingles2)
                    
                    # If we've gone too far, revert and try again
                    if new_similarity > target_similarity + tolerance:
                        text2_chars[pos2:pos2+section_length] = original_section
                        continue
                    
                    # Update our text and similarity
                    text2 = new_text2
                    shingles2 = new_shingles2
                    current_similarity = new_similarity
                    
                    # If we're within tolerance, we're done
                    if abs(current_similarity - target_similarity) <= tolerance:
                        return text1, text2, current_similarity
        
        # If we need to decrease similarity, replace shared shingles
        elif current_similarity > target_similarity:
            # Generate entirely new text and try again
            continue
        
        # Track the best attempt
        difference = abs(current_similarity - target_similarity)
        if difference < best_difference:
            best_difference = difference
            best_text2 = text2
            best_similarity = current_similarity
            
            # If we're within tolerance, we're done
            if difference <= tolerance:
                return text1, text2, current_similarity
    
    # If we couldn't get within tolerance, return our best attempt
    print(f"Warning: Could not achieve exact target similarity. Best: {best_similarity:.4f}")
    return text1, best_text2, best_similarity

def main():
    parser = argparse.ArgumentParser(description='Generate two texts with a target Jaccard similarity')
    parser.add_argument('--target', type=float, default=0.5, help='Target Jaccard similarity (0.0-1.0)')
    parser.add_argument('--shingle-size', type=int, default=3, help='Size of character shingles (k)')
    parser.add_argument('--output1', type=str, default='text1.txt', help='Output file for first text')
    parser.add_argument('--output2', type=str, default='text2.txt', help='Output file for second text')
    parser.add_argument('--tolerance', type=float, default=0.02, help='Acceptable deviation from target similarity')
    parser.add_argument('--verify', action='store_true', help='Verify similarity with C++ algorithm')
    
    args = parser.parse_args()
    
    if args.target < 0 or args.target > 1:
        print("Error: Target similarity must be between 0.0 and 1.0")
        return
    
    text1, text2, actual_similarity = generate_texts_with_target_similarity(
        args.target, args.shingle_size, args.tolerance
    )
    
    # Print sample of each text
    print("\nSample of text1 (first 100 chars):")
    print(text1[:100] + "...")
    print("\nSample of text2 (first 100 chars):")
    print(text2[:100] + "...")
    
    # Count shingles in each text
    shingles1 = get_character_shingles(text1, args.shingle_size)
    shingles2 = get_character_shingles(text2, args.shingle_size)
    
    print(f"\nText1 unique {args.shingle_size}-shingles: {len(shingles1)}")
    print(f"Text2 unique {args.shingle_size}-shingles: {len(shingles2)}")
    print(f"Shared shingles: {len(shingles1.intersection(shingles2))}")
    print(f"Union of shingles: {len(shingles1.union(shingles2))}")
    
    with open(args.output1, 'w') as f1:
        f1.write(text1)
    
    with open(args.output2, 'w') as f2:
        f2.write(text2)
    
    print(f"\nGenerated two texts with Jaccard similarity: {actual_similarity:.4f}")
    print(f"Target similarity was: {args.target:.4f}")
    print(f"Text 1 saved to: {args.output1}")
    print(f"Text 2 saved to: {args.output2}")
    print(f"Character shingle size used: {args.shingle_size}")
    
    if args.verify:
        # You could add code here to call your C++ program to verify
        print("\nTo verify with your C++ program, run:")
        print(f"./jaccard_similarity {args.output1} {args.output2} {args.shingle_size}")

if __name__ == "__main__":
    main()