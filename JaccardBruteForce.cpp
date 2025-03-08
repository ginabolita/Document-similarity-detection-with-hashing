#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>

// Function to generate k-shingles from text
std::unordered_set<std::string> generateShingles(const std::string& text, int k) {
    std::unordered_set<std::string> shingles;
    
    // Generate k-shingles (substrings of length k)
    if (text.length() >= k) {
        for (size_t i = 0; i <= text.length() - k; i++) {
            std::string shingle = text.substr(i, k);
            shingles.insert(shingle);
        }
    }
    
    return shingles;
}

// Calculate Jaccard similarity
double calculateJaccardSimilarity(const std::unordered_set<std::string>& set1, 
                                 const std::unordered_set<std::string>& set2) {
    // Find intersection size
    int intersection = 0;
    for (const auto& shingle : set1) {
        if (set2.find(shingle) != set2.end()) {
            intersection++;
        }
    }
    
    // Calculate union size: |A| + |B| - |A ∩ B|
    int unionSize = set1.size() + set2.size() - intersection;
    
    // Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
    return unionSize > 0 ? static_cast<double>(intersection) / unionSize : 0.0;
}

// Read content from file
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return "";
    }
    
    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + " ";
    }
    
    return content;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <file1> <file2> <k>" << std::endl;
        std::cout << "where k is the shingle size" << std::endl;
        return 1;
    }
    
    // Get k value from command line
    int k = std::stoi(argv[3]);
    if (k <= 0) {
        std::cerr << "Error: k must be positive" << std::endl;
        return 1;
    }
    
    // Read input files
    std::string text1 = readFile(argv[1]);
    std::string text2 = readFile(argv[2]);
    
    if (text1.empty() || text2.empty()) {
        std::cerr << "Error: One or both input files are empty or could not be read." << std::endl;
        return 1;
    }
    
    // Generate k-shingles for both documents
    std::unordered_set<std::string> shingles1 = generateShingles(text1, k);
    std::unordered_set<std::string> shingles2 = generateShingles(text2, k);
    
    // Calculate Jaccard similarity
    double similarity = calculateJaccardSimilarity(shingles1, shingles2);
    
    // Output results
    std::cout << "Number of unique shingles in document 1: " << shingles1.size() << std::endl;
    std::cout << "Number of unique shingles in document 2: " << shingles2.size() << std::endl;
    std::cout << "Jaccard similarity: " << similarity << std::endl;
    
    return 0;
}