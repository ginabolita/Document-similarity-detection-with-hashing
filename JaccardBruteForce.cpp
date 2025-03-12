#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <sstream>

#include <nlohmann/json.hpp> 
using namespace nlohmann;

typedef unsigned int uint;
std::unordered_set<std::string> stopwords;



//StopWord Zone --------------------------------------------------------------------------

bool is_stopword(const std::string& word) {
    return stopwords.find(word) != stopwords.end();
}

std::unordered_set<std::string> loadStopwords(const std::string& filename) {
    std::unordered_set<std::string> stopwords;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return stopwords;
    }

    json j;
    file >> j;  // Parse JSON

    for (const auto& word : j) {
        stopwords.insert(word);
    }

    return stopwords;
}
//-----------------------------------------------------------------------------------





// Format Zone //--------------------------------------------------------------------

std::string remove_punctuation(std::string text)
{
    std::string newtext;
    for (int i = 0; i < text.size(); ++i)
    {
        if (text[i] == '.' || text[i] == ',' || text[i] == '!' || text[i] == '?' || text[i] == ';' || text[i] == ':')
        {
            newtext += ' ';
        }
        else
        {
            newtext += text[i];
        }
    }
    return newtext;
}

std::string normalize(const std::string& word) {
    std::string result;
    result.reserve(word.length());
    for (char c : word) {
        if (isalpha(c)) {
            result += tolower(c);
        }
    }
    return result;
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
//-----------------------------------------------------------------------------------



// Algorithm Zone -------------------------------------------------------------------

// Function to generate k-shingles from text
std::unordered_set<std::string> generateShingles(const std::string& text, uint k) {
    std::unordered_set<std::string> shingles;
    std::vector<std::string> words;
    std::stringstream ss(text);
    std::string word;
    
    // Tokenize the text into words
    while (ss >> word)
    {
        //tenim en compte si es una stopword abans de tot
        if (!is_stopword(normalize(word))){
            words.push_back(normalize(word));
        }

    }
    
    // Generate k-word shingles
    if (words.size() >= k) {
        for (size_t i = 0; i <= words.size() - k; i++) {
            std::string shingle;
            for (size_t j = 0; j < k; j++) {
                if (j > 0) shingle += " "; // Separate words with a space
                shingle += words[i + j];
            }
            shingles.insert(shingle);
        }
    }
    
    //print shingles
    for (auto shingle : shingles){
        std::cout << shingle << " ";
    }
    std::cout << std::endl;


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

//-----------------------------------------------------------------------------------



int main(int argc, char* argv[]) {

    stopwords = loadStopwords("stopwords-ca.json");
    
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

    text1 = remove_punctuation(text1);
    text2 = remove_punctuation(text2);
    
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
