#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "deps/nlohmann/json.hpp"
#include "deps/xxhash/xxhash.h"

using namespace std;
using namespace nlohmann;
unsigned int k; // Tamaño de los k-shingles
unsigned int
    t; // Numero de funciones hash para el minhash (replaces numHashFunctions)
float SIMILARITY_THRESHOLD;

vector<pair<int, int>>
    hashCoefficients;                // [a, b] for funcionhash(x) = (ax + b) % p
int p;                               // Prime number for hash functions
unordered_set<string> stopwords;     // Stopwords
vector<pair<int, int>> similarPairs; // Similar pairs of documents
vector<vector<float>> Data;
map<string, int> timeResults; // Map to store execution times

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
// Performance Measurement <- Marcel, el timer para y mide el tiempo
// automaticamente cuando se destruye
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
    auto duration =
        chrono::duration_cast<chrono::milliseconds>(endTime - startTime)
            .count();
    timeResults[operationName] = duration;
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

  try
  {
    json j;
    file >> j; // Parse JSON

    for (const auto &word : j)
    {
      stopwords.insert(word.get<string>());
    }
    if (stopwords.empty())
    {
      cerr << "Warning: Stopwords file could not be loaded or is empty."
           << endl;
    }
  }
  catch (const exception &e)
  {
    cerr << "Error parsing stopwords file: " << e.what() << endl;
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
          str.find(".doc") != string::npos || str.find(".md") != string::npos);
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
  // A larger prime number for better distribution, but smaller than INT_MAX to
  // avoid overflow
  p = nextPrime(INT_MAX / 4);

  // Use a time-based seed for better randomness
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  mt19937 gen(seed); // Using Mersenne Twister
  uniform_int_distribution<int> dis(1, p - 1);

  hashCoefficients.clear(); // Clear existing coefficients
  hashCoefficients.reserve(t);
  for (unsigned int i = 0; i < t; i++)
  {
    hashCoefficients.push_back({dis(gen), dis(gen)}); // {Random a, Random b}
  }

  if (hashCoefficients.empty())
  {
    cerr << "Error: Hash functions were not initialized." << endl;
    exit(1); // Critical error, exit program
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
    if (!word.empty() &&
        !is_stopword(word))
    { // Skip empty words and stopwords
      palabras.push(word);
      if (palabras.size() ==
          k)
      { // si ya tenemos k palabras en la cola tenemos un k shingle!
        // Optimización: construir el shingle directamente con un string
        // estimado
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
  // Check if there are no shingles
  if (kShingles.empty())
  {
    cerr << "Warning: Empty kShingles set. Creating default signature." << endl;
    return vector<int>(t, INT_MAX); // Return default signature
  }

  vector<int> signature(t, INT_MAX);

  // For each shingle in the set
  for (const string &shingle : kShingles)
  {
    // Use xxHash for better distribution and performance
    uint64_t shingleID = xxHashFunction(shingle);

    // Apply each hash function
    for (unsigned int i = 0; i < t && i < hashCoefficients.size(); i++)
    {
      int a = hashCoefficients[i].first;
      int b = hashCoefficients[i].second;

      // Using modular arithmetic to prevent overflow
      int64_t hashValue = (static_cast<int64_t>(a) * shingleID + b) % p;

      // Ensure positive hash value
      if (hashValue < 0)
        hashValue += p;

      // Update signature with minimum hash value
      signature[i] = min(signature[i], static_cast<int>(hashValue));
    }
  }

  return signature;
}

// Calculate exact Jaccard similarity between two sets of shingles
float exactJaccardSimilarity(const unordered_set<string> &set1,
                             const unordered_set<string> &set2)
{
  // Check for empty sets
  if (set1.empty() && set2.empty())
    return 1.0f; // Both empty = 100% similar
  if (set1.empty() || set2.empty())
    return 0.0f; // One empty = 0% similar

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
  return unionSize > 0 ? static_cast<float>(intersectionSize) / unionSize
                       : 0.0f;
}

// Calculate estimated Jaccard similarity using MinHash signatures
float estimatedJaccardSimilarity(const vector<int> &signature1,
                                 const vector<int> &signature2)
{
  // Check for empty or mismatched signatures
  if (signature1.empty() || signature2.empty() ||
      signature1.size() != signature2.size())
  {
    cerr << "Warning: Invalid signatures for comparison." << endl;
    return 0.0f;
  }

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
    hashValue ^=
        hash<int>{}(value) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
  }
  return hashValue;
}

// Initialize LSH buckets
void initializeLSHBuckets(int numBands)
{
  bandBucketMap.clear();
  bandBucketMap.resize(numBands);
  // cout << "Initialized " << numBands << " LSH bands" << endl;
}

// Add a document to LSH buckets
void addToLSHBuckets(const vector<int> &signature, int docIndex, int numBands)
{
  // Check for empty signature or invalid inputs
  if (signature.empty())
  {
    cerr << "Warning: Empty signature for document " << docIndex << endl;
    return;
  }

  if (numBands <= 0)
  {
    cerr << "Error: Invalid number of bands: " << numBands << endl;
    return;
  }

  if (docIndex < 0)
  {
    cerr << "Error: Invalid document index: " << docIndex << endl;
    return;
  }

  // Calculate band size (rows per band)
  int rowsPerBand = max(1, static_cast<int>(signature.size() / numBands));

  // For each band
  for (int b = 0; b < numBands && b < static_cast<int>(bandBucketMap.size());
       b++)
  {
    // Extract the band (sub-signature)
    vector<int> band;
    int startIdx = b * rowsPerBand;
    int endIdx = min((b + 1) * rowsPerBand, static_cast<int>(signature.size()));

    // Check bounds
    if (startIdx >= static_cast<int>(signature.size()))
    {
      continue; // Skip this band if out of bounds
    }

    band.assign(signature.begin() + startIdx, signature.begin() + endIdx);

    // Hash the band
    size_t bandHash = hashBand(band);

    // Add the document to the corresponding bucket
    bandBucketMap[b][bandHash].docIndices.push_back(docIndex);
  }
}

vector<pair<int, int>> findSimilarDocumentPairs(
    const vector<Document> &documents, int numBands, float threshold)
{
  // cout << "Starting findSimilarDocumentPairs with " << documents.size()
  //      << " documents, " << numBands << " bands, threshold " << threshold
  //      << endl;

  // Validate inputs
  if (documents.empty() || numBands <= 0 ||
      bandBucketMap.size() != static_cast<size_t>(numBands))
  {
    cerr << "Error: Invalid inputs for findSimilarDocumentPairs" << endl;
    return {};
  }

  // Debug: Check if buckets contain any documents
  int totalBuckets = 0;
  int nonEmptyBuckets = 0;
  int maxBucketSize = 0;

  for (int b = 0; b < numBands && b < static_cast<int>(bandBucketMap.size());
       b++)
  {
    totalBuckets += bandBucketMap[b].size();

    for (const auto &bucketPair : bandBucketMap[b])
    {
      const Bucket &bucket = bucketPair.second;
      if (!bucket.docIndices.empty())
      {
        nonEmptyBuckets++;
        maxBucketSize =
            max(maxBucketSize, static_cast<int>(bucket.docIndices.size()));
      }
    }
  }

  // cout << "LSH stats: " << totalBuckets << " total buckets, " << nonEmptyBuckets
  //     << " non-empty buckets, "
  //      << "largest bucket has " << maxBucketSize << " documents" << endl;

  // Set to store pairs of similar documents (to avoid duplicates)
  set<pair<int, int>> similarPairsSet;

  // For each band
  for (int b = 0; b < numBands && b < static_cast<int>(bandBucketMap.size());
       b++)
  {
    // For each bucket in this band
    for (const auto &bucketPair : bandBucketMap[b])
    {
      const Bucket &bucket = bucketPair.second;

      // If bucket has at least 2 documents, they might be similar
      if (bucket.docIndices.size() >= 2)
      {
        // Check all pairs in the bucket
        for (size_t i = 0; i < bucket.docIndices.size(); i++)
        {
          for (size_t j = i + 1; j < bucket.docIndices.size(); j++)
          {
            int doc1 = bucket.docIndices[i];
            int doc2 = bucket.docIndices[j];

            // Check for valid document indices
            if (doc1 < 0 || doc2 < 0 ||
                doc1 >= static_cast<int>(documents.size()) ||
                doc2 >= static_cast<int>(documents.size()))
            {
              continue; // Skip invalid indices
            }

            // Ensure doc1 < doc2 for consistent ordering
            if (doc1 > doc2)
            {
              swap(doc1, doc2);
            }

            similarPairsSet.insert({doc1, doc2});
          }
        }
      }
    }
  }

  // Convert set to vector
  vector<pair<int, int>> candidatePairs(similarPairsSet.begin(),
                                        similarPairsSet.end());
  // cout << "Found " << candidatePairs.size() << " candidate pairs" << endl;

  // Filter pairs based on actual similarity
  vector<pair<int, int>> filteredPairs;
  for (const auto &pair : candidatePairs)
  {
    // Additional index checks
    if (pair.first >= static_cast<int>(documents.size()) ||
        pair.second >= static_cast<int>(documents.size()) || pair.first < 0 ||
        pair.second < 0)
    {
      cerr << "Error: Invalid document indices in pair: " << pair.first << ", "
           << pair.second << endl;
      continue;
    }

    // Check that signatures are valid
    if (documents[pair.first].signature.empty() ||
        documents[pair.second].signature.empty())
    {
      cerr << "Warning: Empty signature(s) for document pair: " << pair.first
           << ", " << pair.second << endl;
      continue;
    }

    float similarity = estimatedJaccardSimilarity(
        documents[pair.first].signature, documents[pair.second].signature);

    if (similarity >= threshold)
    {
      filteredPairs.push_back(pair);
    }
  }

  return filteredPairs;
}

int extractNumber(const std::string &filename)
{
  // Find the last underscore
  size_t underscorePos = filename.find_last_of('_');
  if (underscorePos == std::string::npos)
  {
    return -1; // No underscore found
  }

  // Find the position of the dot after the underscore
  size_t dotPos = filename.find('.', underscorePos);
  if (dotPos == std::string::npos)
  {
    dotPos = filename.length(); // No dot found, use end of string
  }

  // Extract the substring between underscore and dot
  std::string numberStr = filename.substr(underscorePos + 1, dotPos - underscorePos - 1);

  // Convert string to integer
  try
  {
    return std::stoi(numberStr);
  }
  catch (...)
  {
    return -1; // Conversion failed
  }
}

void writeResultsToCSV(const string &filename1,
                       const string &filename2,
                       const vector<pair<int, int>> &similarPairs,
                       const vector<Document> &documents)
{
  // Ensure filename has .csv extension
  string csvFilename = filename1;
  if (csvFilename.substr(csvFilename.length() - 4) != ".csv")
  {
    csvFilename += ".csv";
  }

  ofstream file(csvFilename);
  if (!file.is_open())
  {
    cerr << "Error: Unable to open file " << csvFilename << " for writing" << endl;
    return;
  }

  // Write header
  file << "Document1,Document2,EstimatedSimilarity" << endl;

  // Write data rows
  for (const auto &pair : similarPairs)
  {
    // Extract document IDs from filenames
    string doc1 = documents[pair.first].filename;
    string doc2 = documents[pair.second].filename;

    // Parse document IDs from filenames
    string id1, id2;

    // Extract document number from filename
    int docNum1 = extractNumber(doc1);
    if (docNum1 != -1)
    {
      id1 = to_string(docNum1);
    }
    else
    {
      id1 = doc1;
    }
    int docNum2 = extractNumber(doc2);
    if (docNum2 != -1)
    {
      id2 = to_string(docNum2);
    }
    else
    {
      id2 = doc2;
    }

    // Calculate similarities
    float estSimilarity = estimatedJaccardSimilarity(
        documents[pair.first].signature,
        documents[pair.second].signature);

    // Write to CSV with fixed precision
    file << id1 << ","
         << id2 << ","
         << fixed << setprecision(6) << estSimilarity
         << endl;
  }

  file.close();

  // open new file to store the time results
  string timeFilename = filename2;
  if (timeFilename.substr(timeFilename.length() - 4) != ".csv")
  {
    timeFilename += ".csv";
  }

  ofstream fileTime(timeFilename);
  if (!fileTime.is_open())
  {
    cerr << "Error: Unable to open file TimeResults.csv for writing" << endl;
    return;
  }

  // Write header
  fileTime << "Operation,Time(ms)" << endl;

  for (const auto &pair : timeResults)
  {
    fileTime << pair.first << "," << pair.second << endl;
  }

  fileTime.close();
  cout << "Results written to " << csvFilename << endl;
}

std::string determineCategory(const std::string &inputDirectory)
{
  if (inputDirectory.find("real") != std::string::npos)
  {
    return "real";
  }
  else if (inputDirectory.find("virtual") != std::string::npos)
  {
    return "virtual";
  }
  return "unknown"; // Fallback case
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
void printUsage(const char *programName)
{
  cout << "Usage options:" << endl;
  cout << "1. Compare all files in corpus: " << programName
       << " <corpus_dir> <k> <b> <t> <sim_threshold>" << endl;
  cout << "where:" << endl;
  cout << "  <corpus_dir>: Directory containing text files to compare" << endl;
  cout << "  <k>: Shingle size (number of consecutive words)" << endl;
  cout << "  <b>: Number of bands for LSH" << endl;
  cout << "  <t>: Number of hash functions" << endl;
  cout << "  <sim_threshold>: Similarity threshold (0.0 to 1.0)" << endl;
}

int main(int argc, char *argv[])
{
  // Start global timer for entire execution
  auto startTime = chrono::high_resolution_clock::now();
  string filename2,filename1, category;
  vector<Document> documents;

  { // Check command line arguments
    Timer timerStopwords("Total Execution Time: ");
    if (argc != 6)
    {
      printUsage(argv[0]);
      return 1;
    }

    // Parse parameters
    string corpusDir = argv[1];
    k = stoi(argv[2]);                    // Shingle size
    int b = stoi(argv[3]);                // Number of bands
    t = stoi(argv[4]);                    // Number of hash functions
    SIMILARITY_THRESHOLD = stof(argv[5]); // Similarity threshold

    // Validate inputs
    if (k <= 0 || b <= 0 || t <= 0 || SIMILARITY_THRESHOLD <= 0 ||
        SIMILARITY_THRESHOLD > 1.0)
    {
      cerr << "Error: k, b, t must be positive and similarity threshold must be "
              "between 0 and 1"
           << endl;
      return 1;
    }

    // Check if corpus directory exists
    if (!filesystem::exists(corpusDir))
    {
      cerr << "Error: Directory " << corpusDir << " does not exist" << endl;
      return 1;
    }

    if (!filesystem::is_directory(corpusDir))
    {
      cerr << "Error: " << corpusDir << " is not a directory" << endl;
      return 1;
    }

    // Load stopwords - Handle potential missing file gracefully
    try
    {
      // cout << "Loading stopwords..." << endl;
      stopwords = loadStopwords("stopwords-en.json");
      // cout << "Loaded " << stopwords.size() << " stopwords" << endl;
    }
    catch (const exception &e)
    {
      cerr << "Warning: Error loading stopwords: " << e.what() << endl;
      // Continue execution even if stopwords can't be loaded
    }

    // Initialize hash functions - THIS WAS MISSING IN THE ORIGINAL CODE
    // cout << "Initializing hash functions..." << endl;
    initializeHashFunctions();
    // cout << "Initialized " << hashCoefficients.size() << " hash functions"
    //     << endl;

    // Process all files in the corpus directory
    vector<Document> documents;
    {
      Timer timerProcessCorpus("Processing corpus");
      // cout << "Processing files in directory: " << corpusDir << endl;

      // Count how many files we'll process
      int fileCount = 0;
      for (const auto &entry : filesystem::directory_iterator(corpusDir))
      {
        if (entry.is_regular_file() && isFilePath(entry.path().string()))
        {
          fileCount++;
        }
      }
      // cout << "Found " << fileCount << " files to process" << endl;

      // Process each file
      int processedCount = 0;
      for (const auto &entry : filesystem::directory_iterator(corpusDir))
      {
        if (entry.is_regular_file() && isFilePath(entry.path().string()))
        {
          string filename = entry.path().string();
          Document doc(filename);

          // Read and process file
          string content = readFile(filename);
          if (content.empty())
          {
            cerr << "Warning: File " << filename
                 << " is empty or could not be read. Skipping." << endl;
            continue;
          }

          // Extract shingles
          tratar(content, doc.kShingles);
          if (doc.kShingles.empty())
          {
            cerr << "Warning: No valid shingles extracted from " << filename
                 << ". Skipping." << endl;
            continue;
          }

          // Compute MinHash signature
          doc.signature = computeMinHashSignature(doc.kShingles);

          // Add to documents collection
          documents.push_back(doc);

          // Progress reporting
          processedCount++;
          if (processedCount % 10 == 0 || processedCount == fileCount)
          {
            // cout << "Processed " << processedCount << "/" << fileCount << " files"
            //     << endl;
          }
        }
      }
    }

    // Check if we have enough documents
    if (documents.size() < 2)
    {
      cerr << "Error: Need at least 2 valid documents to compare. Found: "
           << documents.size() << endl;
      return 1;
    }

    // Initialize LSH buckets
    {
      Timer timerInitBuckets("Initializing LSH buckets");
      initializeLSHBuckets(b);
    }

    // Add documents to LSH buckets
    {
      Timer timerLSH("LSH Bucketing");
      for (size_t i = 0; i < documents.size(); i++)
      {
        addToLSHBuckets(documents[i].signature, i, b);
      }
    }

    // Find similar document pairs
    {
      Timer timerFindSimilar("Finding similar documents");
      similarPairs = findSimilarDocumentPairs(documents, b, SIMILARITY_THRESHOLD);
    }

    category = determineCategory(argv[1]);

    // Ensure the category is valid
    if (category == "unknown")
    {
      std::cerr << "Warning: Could not determine category from input directory!" << std::endl;
      return 1;
    }

    std::stringstream ss;
    ss << "results/" << category << "/bucketing/bucketingSimilarities_k" << k
       << "_t" << t
       << "_b" << b
       << "_threshold" << SIMILARITY_THRESHOLD << ".csv";

    filename1 = ss.str();

    // Generate the second filename with the same structure (e.g., for time measurements)
    std::stringstream ss2;
    ss2 << "results/" << category << "/bucketing/bucketingTimes_k" << k
        << "_t" << t
        << "_b" << b
        << "_threshold" << SIMILARITY_THRESHOLD << ".csv";

    filename2 = ss2.str();
  }

  writeResultsToCSV(filename1, filename2, similarPairs, documents);

  // Calculate and display total execution time
  auto endTime = chrono::high_resolution_clock::now();
  auto duration =
      chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
  cout << "time: " << duration.count() << " ms" << endl;

  return 0;
}
