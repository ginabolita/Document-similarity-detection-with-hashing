#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include "deps/nlohmann/json.hpp"
#include "deps/xxhash/xxhash.h"
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <functional>

#define SIMILARITY_THRESHOLD 0.01f // Threshold for considering documents similar

using namespace std;
using namespace nlohmann;
unsigned int k;                          // Tamaño de los k-shingles
unsigned int t = 100;                    // Numero de funciones hash para el minhash (replaces numHashFunctions)
vector<pair<int, int>> hashCoefficients; // [a, b] for funcionhash(x) = (ax + b) % p
int p;                                   // Prime number for hash functions
unordered_set<string> stopwords;         // Stopwords
vector<pair<int, int>> similarPairs;     // Similar pairs of documents

// Document structure to store document information
struct Document
{
  string filename;
  unordered_set<string> kShingles;
  vector<int> signature;

  Document(const string &name) : filename(name) {}
};

// Bucket structure for LSH
struct Bucket
{
  vector<int> docIndices; // Indices of documents in this bucket
};

// LSH structure to store buckets by band
typedef unordered_map<size_t, Bucket> BandBuckets;
vector<BandBuckets> bandBucketMap;

//---------------------------------------------------------------------------
// Performance Measurement <- Marcel, el timer para y mide el tiempo automaticamente cuando se destruye
//---------------------------------------------------------------------------
class Timer
{
private:
  chrono::high_resolution_clock::time_point startTime;
  string operationName;

public:
  Timer(const string &name) : operationName(name)
  {
    startTime = chrono::high_resolution_clock::now();
  }

  ~Timer()
  {
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    //cout << "[Performance] " << operationName << ": " << duration << " ms" << endl;
  }
};

//---------------------------------------------------------------------------
// Treating StopWords
// --------------------------------------------------------------------------
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

//---------------------------------------------------------------------------
// Treating Format
//---------------------------------------------------------------------------

// Quita signos de puntuacion y mayusculas
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

// Read content from file
string readFile(const string &filename)
{
  ifstream file(filename);
  if (!file.is_open())
  {
    cerr << "Error opening file: " << filename << endl;
    return "";
  }

  string content;
  string line;
  while (getline(file, line))
  {
    content += line + " ";
  }

  return content;
}

// Function to check if a string is a file path
bool isFilePath(const string &str)
{
  return (str.find(".txt") != string::npos ||
          str.find(".doc") != string::npos ||
          str.find(".md") != string::npos);
}

// Optimized function to find the next prime after n
int nextPrime(int n)
{
  // Handle even numbers
  if (n <= 2)
    return 2;
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

    if (isPrime)
      return n;
    n += 2; // Skip even numbers in search
  }
}

//---------------------------------------------------------------------------
// Improved Hash Functions
//---------------------------------------------------------------------------

// xxHash implementation for better hash performance
size_t xxHashFunction(const string &str, uint64_t seed = 0)
{
  return XXH64(str.c_str(), str.length(), seed);
}

// Initialize hash functions with random coefficients
void initializeHashFunctions()
{
  p = nextPrime(INT_MAX / 2); // A larger prime number for better distribution

  // Use a time-based seed for better randomness
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  mt19937_64 gen(seed); // Using 64-bit Mersenne Twister
  uniform_int_distribution<int64_t> dis(1, p - 1);

  hashCoefficients.reserve(t);
  for (unsigned int i = 0; i < t; i++)
  {
    hashCoefficients.push_back({dis(gen), dis(gen)}); // {Random a, Random b}
  }
}

//---------------------------------------------------------------------------
// Jaccard Locality-Sensitive Hashing Algorithm
//---------------------------------------------------------------------------

// Function to process text and extract k-shingles
void tratar(const string &texto, unordered_set<string> &kShingles)
{
  queue<string> palabras; // cola para tener las k palabras consecutivas
  string word;
  stringstream ss(texto);

  while (ss >> word)
  { // leer palabra del stringstream
    // quitar singnos de puntuacion y mayusculas
    word = normalize(word);
    if (!word.empty() && !is_stopword(word))
    { // Skip empty words and stopwords
      palabras.push(word);
      if (palabras.size() == k)
      { // si ya tenemos k palabras en la cola tenemos un k shingle!
        // Optimización: construir el shingle directamente con un string estimado
        string shingle;
        shingle.reserve(k * 10); // Reservar espacio aproximado

        queue<string> temp = palabras;
        for (size_t i = 0; i < k; i++)
        {
          shingle += temp.front();
          temp.pop();
          if (i < k - 1)
            shingle += " ";
        }

        kShingles.insert(shingle);
        // Quitamos la primera para avanzar (sliding window approach)
        palabras.pop();
      }
    }
  }
}

// Improved function to compute MinHash signatures using xxHash
vector<int> computeMinHashSignature(const unordered_set<string> &kShingles)
{
  vector<int> signature(t, INT_MAX);

  // For each shingle in the set
  for (const string &shingle : kShingles)
  {
    // Use xxHash for better distribution and performance
    uint64_t shingleID = xxHashFunction(shingle);

    // Apply each hash function
    for (unsigned int i = 0; i < t; i++)
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

// Calculate exact Jaccard similarity between two sets of shingles
float exactJaccardSimilarity(const unordered_set<string> &set1, const unordered_set<string> &set2)
{
  // Count intersection size
  size_t intersectionSize = 0;
  for (const auto &shingle : set1)
  {
    if (set2.find(shingle) != set2.end())
    {
      intersectionSize++;
    }
  }

  // Calculate union size: |A| + |B| - |A∩B|
  size_t unionSize = set1.size() + set2.size() - intersectionSize;

  // Return Jaccard similarity
  return unionSize > 0 ? static_cast<float>(intersectionSize) / unionSize : 0.0f;
}

// Calculate estimated Jaccard similarity using MinHash signatures
float estimatedJaccardSimilarity(const vector<int> &signature1, const vector<int> &signature2)
{
  int matchingElements = 0;

  // Count positions where signatures match
  for (size_t i = 0; i < signature1.size(); i++)
  {
    if (signature1[i] == signature2[i])
    {
      matchingElements++;
    }
  }

  return static_cast<float>(matchingElements) / signature1.size();
}

// Create a hash for a band (sub-signature)
size_t hashBand(const vector<int> &band)
{
  size_t hashValue = 0;
  for (int value : band)
  {
    // Combine hash values - similar to boost::hash_combine
    hashValue ^= hash<int>{}(value) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
  }
  return hashValue;
}

// Initialize LSH buckets
void initializeLSHBuckets(int numBands)
{
  bandBucketMap.clear();
  bandBucketMap.resize(numBands);
  //cout << "Initialized " << numBands << " LSH bands" << endl;
}

// Add a document to LSH buckets
void addToLSHBuckets(const vector<int> &signature, int docIndex, int numBands)
{
  // Calculate band size (rows per band)
  int rowsPerBand = signature.size() / numBands;
  
  if (rowsPerBand == 0) {
    rowsPerBand = 1;  // Ensure at least one row per band
  }

  //cout << "Adding document " << docIndex << " to LSH buckets (signature size: " 
   //    << signature.size() << ", rows per band: " << rowsPerBand << ")" << endl;

  int bucketsAdded = 0;
  
  // For each band
  for (int b = 0; b < numBands; b++)
  {
    // Extract the band (sub-signature)
    vector<int> band;
    int startIdx = b * rowsPerBand;
    int endIdx = min((b + 1) * rowsPerBand, static_cast<int>(signature.size()));

    band.assign(signature.begin() + startIdx, signature.begin() + endIdx);

    // Hash the band
    size_t bandHash = hashBand(band);

    // Add the document to the corresponding bucket
    bandBucketMap[b][bandHash].docIndices.push_back(docIndex);
    bucketsAdded++;
  }
  
  //cout << "Document " << docIndex << " added to " << bucketsAdded << " buckets" << endl;
}

vector<pair<int, int>> findSimilarDocumentPairs(const vector<Document> &documents, int numBands, float threshold)
{
  //cout << "Starting findSimilarDocumentPairs with " << documents.size() 
   //    << " documents, " << numBands << " bands, threshold " << threshold << endl;
  
  // Debug: Check if buckets contain any documents
  int totalBuckets = 0;
  int nonEmptyBuckets = 0;
  int maxBucketSize = 0;
  
  for (int b = 0; b < numBands; b++)
  {
    totalBuckets += bandBucketMap[b].size();
    
    for (const auto &bucketPair : bandBucketMap[b])
    {
      const Bucket &bucket = bucketPair.second;
      if (!bucket.docIndices.empty()) {
        nonEmptyBuckets++;
        maxBucketSize = max(maxBucketSize, static_cast<int>(bucket.docIndices.size()));
      }
    }
  }
  
  //cout << "LSH stats: " << totalBuckets << " total buckets, " 
  //     << nonEmptyBuckets << " non-empty buckets, "
   //    << "largest bucket has " << maxBucketSize << " documents" << endl;

  // Set to store pairs of similar documents (to avoid duplicates)
  set<pair<int, int>> similarPairsSet;


  // For each band
  for (int b = 0; b < numBands; b++)
  {
    //cout << "Processing band " << b << " with " << bandBucketMap[b].size() << " buckets" << endl;
    
    // For each bucket in this band
    for (const auto &bucketPair : bandBucketMap[b])
    {
      const Bucket &bucket = bucketPair.second;

      // If bucket has at least 2 documents, they might be similar
      if (bucket.docIndices.size() >= 2)
      {
        //cout << "Found bucket with " << bucket.docIndices.size() << " documents" << endl;
        
        // Check all pairs in the bucket
        for (size_t i = 0; i < bucket.docIndices.size(); i++)
        {
          for (size_t j = i + 1; j < bucket.docIndices.size(); j++)
          {
            int doc1 = bucket.docIndices[i];
            int doc2 = bucket.docIndices[j];

            // Ensure doc1 < doc2 for consistent ordering
            if (doc1 > doc2)
            {
              swap(doc1, doc2);
            }

            similarPairsSet.insert({doc1, doc2});
            
            // Debug: Output when a pair is added
            //cout << "Added candidate pair: " << doc1 << " and " << doc2 << endl;
          }
        }
      }
    }
  }

  // Convert set to vector
  vector<pair<int, int>> similarPairs(similarPairsSet.begin(), similarPairsSet.end());
  
  cout << "Found " << similarPairs.size() << " candidate pairs" << endl;

  // Filter pairs based on actual similarity
  vector<pair<int, int>> filteredPairs;
  for (const auto &pair : similarPairs)
  {
    //cout << "Comparing documents: " << documents[pair.first].filename << " and " << documents[pair.second].filename << endl;
    float similarity = estimatedJaccardSimilarity(
        documents[pair.first].signature,
        documents[pair.second].signature);
    
    //cout << "Estimated Similarity: " << similarity << endl;

    if (similarity >= threshold)
    {
      filteredPairs.push_back(pair);
    }
  }

  return filteredPairs;
}



//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
void printUsage(const char *programName)
{
  cout << "Usage options:" << endl;
  cout << "1. Compare two files: " << programName << " <file1> <file2> <k> <b>" << endl;
  cout << "2. Compare one file with corpus: " << programName << " <file> <corpus_dir> <k> <b>" << endl;
  cout << "3. Compare all files in corpus: " << programName << " <corpus_dir> <k> <b>" << endl;
  cout << "where k is the shingle size and b is the number of bands" << endl;
}

int main(int argc, char *argv[])
{
  // Load stopwords
  stopwords = loadStopwords("stopwords-en.json");

  // Check command line arguments
  if (argc < 4 || argc > 5)
  {
    printUsage(argv[0]);
    return 1;
  }

  // Parse common parameters
  int paramOffset = 0;
  string path1, path2;
  bool corpusMode = false;
  bool singleVsCorpusMode = false;

  // Determine mode based on arguments
  if (argc == 4)
  { // Corpus mode: program <corpus_dir> <k> <b>
    path1 = argv[1];
    if (!filesystem::is_directory(path1))
    {
      cerr << "Error: " << path1 << " is not a directory" << endl;
      return 1;
    }
    corpusMode = true;
    paramOffset = 1;
  }
  else if (argc == 5)
  { // Two files or one file vs corpus
    path1 = argv[1];
    path2 = argv[2];

    if (filesystem::is_directory(path2))
    {
      singleVsCorpusMode = true;
    }
    else if (!filesystem::is_regular_file(path1) || !filesystem::is_regular_file(path2))
    {
      cerr << "Error: One or both paths are not valid files" << endl;
      return 1;
    }
    paramOffset = 2;
  }

  // Get k value from command line
  k = stoi(argv[1 + paramOffset]);
  if (k <= 0)
  {
    cerr << "Error: k must be positive" << endl;
    return 1;
  }

  // Get b value from command line
  int b = stoi(argv[2 + paramOffset]);
  if (b <= 0)
  {
    cerr << "Error: b must be positive" << endl;
    return 1;
  }

  // Adjust number of bands based on threshold
  //cout << "Using " << b << " bands with threshold " << SIMILARITY_THRESHOLD << endl;

  // If threshold is very low, suggest using more bands
  if (SIMILARITY_THRESHOLD < 0.1 && b < 50) {
    cout << "Warning: For low threshold (" << SIMILARITY_THRESHOLD 
        << "), consider using more bands (current: " << b << ")" << endl;
  }


  //bloque de codigo para que al finalizar se destruya el timer (y mida el tiempo automaticamente)
  { // Initialize hash functions
    Timer timerInit("Initialize hash functions");
    initializeHashFunctions();
  }

  // Process documents
  vector<Document> documents;

  if (corpusMode)
  {
    // Process all files in corpus directory
    {
      Timer timerProcessCorpus("Processing corpus");
      cout << "Processing files in directory: " << path1 << endl;

      for (const auto &entry : filesystem::directory_iterator(path1))
      {
        if (entry.is_regular_file() && isFilePath(entry.path().string()))
        {
          string filename = entry.path().string();
          Document doc(filename);

          // Read and process file
          string content = readFile(filename);
          tratar(content, doc.kShingles);

          // Compute MinHash signature
          doc.signature = computeMinHashSignature(doc.kShingles);

          documents.push_back(doc);
          //cout << "Processed: " << filename << " - " << doc.kShingles.size() << " shingles" << endl;
        }
      }
    }

    // Initialize LSH buckets
    initializeLSHBuckets(b);

    {
      // Add documents to LSH buckets
      Timer timerLSH("LSH Bucketing");
      for (size_t i = 0; i < documents.size(); i++)
      {
        addToLSHBuckets(documents[i].signature, i, b);
      }
    }

    {
      // Find similar document pairs
      Timer timerFindSimilar("Finding similar documents");
      similarPairs = findSimilarDocumentPairs(documents, b, SIMILARITY_THRESHOLD);
    }

    // Report results
    cout << "\nFound " << similarPairs.size() << " similar document pairs:" << endl;
    for (const auto &pair : similarPairs)
    {
      float estSimilarity = estimatedJaccardSimilarity(
          documents[pair.first].signature,
          documents[pair.second].signature);

      float exactSimilarity = exactJaccardSimilarity(
          documents[pair.first].kShingles,
          documents[pair.second].kShingles);

      cout << "Similar documents:" << endl;
      cout << "  - " << documents[pair.first].filename << endl;
      cout << "  - " << documents[pair.second].filename << endl;
      cout << "  - Estimated similarity: " << estSimilarity << endl;
      cout << "  - Exact similarity: " << exactSimilarity << endl;
      cout << endl;
    }
  }
  else if (singleVsCorpusMode)
  {
    // Single vs corpus mode
    Document queryDoc(path1);
    vector<pair<float, int>> similarities;

    {
      Timer timerProcessFile("Processing query file");
      cout << "Processing query file: " << path1 << endl;

      string content = readFile(path1);
      tratar(content, queryDoc.kShingles);
      queryDoc.signature = computeMinHashSignature(queryDoc.kShingles);

      //cout << "Processed: " << path1 << " - " << queryDoc.kShingles.size() << " shingles" << endl;
    }

    {
      Timer timerProcessCorpus("Processing corpus");
      cout << "Processing corpus directory: " << path2 << endl;

      for (const auto &entry : filesystem::directory_iterator(path2))
      {
        if (entry.is_regular_file() && isFilePath(entry.path().string()))
        {
          string filename = entry.path().string();
          Document doc(filename);

          string content = readFile(filename);
          tratar(content, doc.kShingles);
          doc.signature = computeMinHashSignature(doc.kShingles);

          documents.push_back(doc);
          //cout << "Processed: " << filename << " - " << doc.kShingles.size() << " shingles" << endl;
        }
      }
    }

    {
      Timer timerFindSimilar("Finding similar documents");

      for (size_t i = 0; i < documents.size(); i++)
      {
        float similarity = estimatedJaccardSimilarity(queryDoc.signature, documents[i].signature);
        if (similarity >= SIMILARITY_THRESHOLD)
        {
          similarities.push_back({similarity, i});
        }
      }

      // Sort by similarity (descending)
      sort(similarities.begin(), similarities.end(),
           [](const pair<float, int> &a, const pair<float, int> &b)
           {
             return a.first > b.first;
           });
    }

    // Report results
    cout << "\nFound " << similarities.size() << " similar documents to " << path1 << ":" << endl;
    for (const auto &pair : similarities)
    {
      float estSimilarity = pair.first;
      int docIndex = pair.second;

      float exactSimilarity = exactJaccardSimilarity(
          queryDoc.kShingles,
          documents[docIndex].kShingles);

      cout << "Similar document:" << endl;
      cout << "  - " << documents[docIndex].filename << endl;
      cout << "  - Estimated similarity: " << estSimilarity << endl;
      cout << "  - Exact similarity: " << exactSimilarity << endl;
      cout << endl;
    }
  }
  else
  {
    // For comparing two files
    Document doc1(path1);
    Document doc2(path2);
    float estSimilarity;
    float exactSimilarity;

    {
      Timer timerProcessFiles("Processing files");
      cout << "Comparing two files: " << path1 << " and " << path2 << endl;

      string content1 = readFile(path1);
      string content2 = readFile(path2);

      tratar(content1, doc1.kShingles);
      tratar(content2, doc2.kShingles);

      doc1.signature = computeMinHashSignature(doc1.kShingles);
      doc2.signature = computeMinHashSignature(doc2.kShingles);

      //cout << "Processed: " << path1 << " - " << doc1.kShingles.size() << " shingles" << endl;
      //cout << "Processed: " << path2 << " - " << doc2.kShingles.size() << " shingles" << endl;
    }

    {
      Timer timerCalcSimilarity("Calculating similarity");
      estSimilarity = estimatedJaccardSimilarity(doc1.signature, doc2.signature);
      exactSimilarity = exactJaccardSimilarity(doc1.kShingles, doc2.kShingles);
    }

    // Report results
    cout << "\nSimilarity between files:" << endl;
    cout << "  - " << path1 << endl;
    cout << "  - " << path2 << endl;
    cout << "  - Estimated similarity (MinHash): " << estSimilarity << endl;
    cout << "  - Exact similarity (Jaccard): " << exactSimilarity << endl;
  }

  return 0;
}