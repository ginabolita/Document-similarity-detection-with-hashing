#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <queue>
#include <sstream>
#include <random>
#include <climits>
#include <cctype>
#include <fstream>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <regex>
#include "deps/nlohmann/json.hpp"


using namespace std;
using namespace nlohmann;
namespace fs = std::filesystem;

unsigned int k;                          // Size of k-shingles
int numHashFunctions;                    // Number of hash functions for minhash (now configurable)
vector<pair<int, int>> hashCoefficients; // [a, b] for hashFunction(x) = (ax + b) % p
int p;                                   // Prime number for hash functions
unordered_set<string> stopwords;         // Stopwords


int extractNumber(const std::string& filename) {
    std::regex pattern(R"(docExp2_(\d+)\.txt)");
    std::smatch match;
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1]);
    }
    return -1; // Si no se encuentra un número, devuelve -1 o maneja el error como prefieras

}

// StopWordsZone ------------------------------------------------------------------------

// Check if a word is a stopword
bool is_stopword(const string &word)
{
    return stopwords.find(word) != stopwords.end();
}

// load stopwords from a file into stopword set
unordered_set<string> loadStopwords(const string &filename)
{
    unordered_set<string> stopwords;
    ifstream file(filename);
    if (!file)
    {
        cerr << "Error opening file: " << filename << endl;
        return stopwords;
    }

    json j;
    file >> j; // Parse JSON

    for (const auto &word : j)
    {
        stopwords.insert(word.get<string>());
    }

    return stopwords;
}

//-----------------------------------------------------------------------------------------


//Format Zone ----------------------------------------------------------------------------

// Remove punctuation and convert to lowercase
string normalize(const string &word)
{
    string result;
    result.reserve(word.length());
    for (char c : word)
    {
        if (isalpha(c))
        {
            result += tolower(c);
        }
    }
    return result;
}

// Function to read text from a file
string readFromFile(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return "";
    }

    stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    return buffer.str();
}

// Read content from file
std::string readFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return "";
    }

    std::string content;
    std::string line;
    while (std::getline(file, line))
    {
        content += line + " ";
    }

    return content;
}

// Optimized function to find the next prime after n
int nextPrime(int n)
{
    // Handle even numbers
    if (n % 2 == 0)
        n++;

    while (true)
    {
        bool isPrime = true;
        // Only need to check up to sqrt(n)
        int sqrtN = sqrt(n);

        // Start from 3 and check only odd numbers
        for (int i = 3; i <= sqrtN; i += 2)
        {
            if (n % i == 0)
            {
                isPrime = false;
                break;
            }
        }

        // Check special case for n = 1 or n = 2
        if (n == 1)
            isPrime = false;
        if (n == 2)
            isPrime = true;

        if (isPrime)
            return n;
        n += 2; // Skip even numbers in search
    }
}

//----------------------------------------------------------------------------------------


// Algoritmo Zone ---------------------------------------------------------------------------

// Initialize hash functions with random coefficients
void initializeHashFunctions()
{
    p = nextPrime(10000); // A prime number larger than maximum possible shingle ID

    // Use a time-based seed for better randomness
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> dis(1, p - 1);

    hashCoefficients.reserve(numHashFunctions);
    for (int i = 0; i < numHashFunctions; i++)
    {
        hashCoefficients.push_back({dis(gen), dis(gen)}); // {Random a, Random b}
    }
}

// Function to process text and extract k-shingles
void tratar(const string &texto, unordered_set<string> &kShingles)
{
    queue<string> palabras; // queue to hold k consecutive words
    string word;
    stringstream ss(texto);

    while (ss >> word)
    {
        // Remove punctuation and convert to lowercase
        word = normalize(word);
        if (!is_stopword(word))
        {
            if (word.empty())
                continue; // Skip empty words after normalization

            palabras.push(word);
            if (palabras.size() == k)
            { // If we have k words in the queue, we have a k-shingle!
                // Optimization: build the shingle directly with an estimated string size
                string shingle;
                shingle.reserve(k * 10); // Reserve approximate space

                queue<string> temp = palabras;
                for (size_t i = 0; i < k; i++)
                {
                    shingle += temp.front();
                    temp.pop();
                    if (i < k - 1)
                        shingle += " ";
                }

                kShingles.insert(shingle);
                // Remove the first word to advance (sliding window approach)
                palabras.pop();
            }
        }
    }
}

// Function to compute MinHash signatures
vector<int> computeMinHashSignature(const unordered_set<string> &kShingles)
{
    vector<int> signature(numHashFunctions, INT_MAX);

    // Cache hash values to avoid recomputation
    hash<string> hasher;

    // For each shingle in the set
    for (const string &shingle : kShingles)
    {
        int shingleID = hasher(shingle); // Convert shingle to a unique integer

        // Apply each hash function
        for (int i = 0; i < numHashFunctions; i++)
        {
            int a = hashCoefficients[i].first;
            int b = hashCoefficients[i].second;
            int hashValue = (1LL * a * shingleID + b) % p; // Using 1LL to prevent overflow

            // Update signature with minimum hash value
            signature[i] = min(signature[i], hashValue);
        }
    }

    return signature;
}

// Calculate Jaccard similarity
float SimilaridadDeJaccard(const vector<int> &signature1, const vector<int> &signature2)
{
    int iguales = 0;

    // When 2 minhashes are equal at a position, it means the shingle used to calculate
    // that position is the same in both texts
    for (int i = 0; i < numHashFunctions; i++)
    {
        if (signature1[i] == signature2[i])
        {
            iguales++;
        }
    }

    return static_cast<float>(iguales) / numHashFunctions;
}

//-------------------------------------------------------------------------------------------

// Function to check if a file is a text file
bool isTextFile(const string &filename)
{
    string extension = fs::path(filename).extension().string();
    return (extension == ".txt" || extension == ".doc" || extension == ".md" || extension == ".text");
}

int main(int argc, char *argv[])
{
    stopwords = loadStopwords("stopwords-en.json");

    if (argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <directory> <k> <t>" << std::endl;
        std::cout << "where k is the shingle size and t is the number of hash functions" << std::endl;
        return 1;
    }

    string directory = argv[1];
    
    // Get k value from command line
    k = std::stoi(argv[2]);
    if (k <= 0)
    {
        std::cerr << "Error: k must be positive" << std::endl;
        return 1;
    }
    
    // Get numHashFunctions value from command line
    numHashFunctions = std::stoi(argv[3]);
    if (numHashFunctions <= 0)
    {
        std::cerr << "Error: Number of hash functions must be positive" << std::endl;
        return 1;
    }

    // Check if directory exists
    if (!fs::exists(directory) || !fs::is_directory(directory))
    {
        std::cerr << "Error: Directory " << directory << " does not exist" << std::endl;
        return 1;
    }

    // Collect all text files in the directory
    vector<string> files;
    for (const auto &entry : fs::directory_iterator(directory))
    {
        if (entry.is_regular_file() && isTextFile(entry.path().string()))
        {
            files.push_back(entry.path().string());
        }
    }

    if (files.empty())
    {
        std::cerr << "Error: No text files found in directory " << directory << std::endl;
        return 1;
    }

    //std::cout << "Found " << files.size() << " text files in directory" << std::endl;
    
    // Initialize hash functions
    initializeHashFunctions();

    // Process each file and compute MinHash signatures
    vector<pair<string, vector<int>>> signatures;
    vector<pair<string, unordered_set<string>>> shingleSets;

    for (const auto &file : files)
    {
        //std::cout << "Processing file: " << file << std::endl;
        string text = readFile(file);
        
        if (text.empty())
        {
            std::cerr << "Warning: File " << file << " is empty or could not be read. Skipping." << std::endl;
            continue;
        }
        
        unordered_set<string> kShingles;
        size_t estimatedSize = max(1UL, (unsigned long)text.length() / 10);
        kShingles.reserve(estimatedSize);
        
        tratar(text, kShingles);
        
        if (kShingles.empty())
        {
            std::cerr << "Warning: No k-shingles could be extracted from file " << file 
                      << ". Make sure the file has at least " << k << " words. Skipping." << std::endl;
            continue;
        }
        
        vector<int> signature = computeMinHashSignature(kShingles);
        signatures.push_back({file, signature});
        shingleSets.push_back({file, kShingles});
    }

    // Compare all pairs of files
    //std::cout << "\nSimilarity Results:\n" << std::endl;
    std::cout << "Doc1,Doc2,Sim%" << std::endl;
   
    for (size_t i = 0; i < signatures.size(); i++)
    {
        for (size_t j = i + 1; j < signatures.size(); j++)
        {
            float similarity = SimilaridadDeJaccard(signatures[i].second, signatures[j].second);
            
            // Get just the filenames without the full path for better readability
            string fileA = fs::path(signatures[i].first).filename().string();
            string fileB = fs::path(signatures[j].first).filename().string();
            
            int numA = extractNumber(fileA);
            int numB = extractNumber(fileB);

            std::cout << numA << "," << numB << "," << similarity * 100 << std::endl;
        }
    }

    return 0;
}