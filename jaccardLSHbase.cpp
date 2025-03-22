#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include "deps/nlohmann/json.hpp"
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <filesystem> // For directory iteration
#include <map>        // For storing results

using namespace std;
using namespace nlohmann;
namespace fs = std::filesystem;

unsigned int k;                          // Size of k-shingles
int numHashFunctions;                    // Number of hash functions for minhash (now a variable)
vector<pair<int, int>> hashCoefficients; // [a, b] for hash function(x) = (ax + b) % p
int p;                                   // Prime number for hash functions
unordered_set<string> stopwords;         // Stopwords
vector<vector<float>> Data;
map<string, int> timeResults; // Map to store execution times

// Struct to hold similarity results
struct SimilarityResult
{
  string file1;
  string file2;
  float similarity;
  bool isSimilar;
};

// Timer class to measure execution time
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

  json j;
  file >> j; // Parse JSON

  for (const auto &word : j)
  {
    stopwords.insert(word);
  }

  return stopwords;
}

//---------------------------------------------------------------------------
// Treating Format
//---------------------------------------------------------------------------

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

//---------------------------------------------------------------------------
// Jaccard Locality-Sensitive Hashing Algorithm
//---------------------------------------------------------------------------

// Initialize hash functions with random coefficients
void initializeHashFunctions()
{
  p = nextPrime(
      10000); // A prime number larger than maximum possible shingle ID

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
  { // read word from stringstream
    // remove punctuation and convert to lowercase
    word = normalize(word);
    if (!is_stopword(word))
    {
      if (word.empty())
        continue; // Skip empty words after normalization

      palabras.push(word);
      if (palabras.size() == k)
      { // if we have k words in the queue, we have a k-shingle!
        // Optimization: build the shingle directly with an estimated string
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
      int hashValue =
          (1LL * a * shingleID + b) % p; // Using 1LL to prevent overflow

      // Update signature with minimum hash value
      signature[i] = min(signature[i], hashValue);
    }
  }

  return signature;
}

float SimilaridadDeJaccard(const vector<int> &signature1,
                           const vector<int> &signature2)
{
  int iguales = 0; // Changed to int for optimization

  // When 2 minhashes are equal in one position, it means the shingle used to calculate
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

// Function to extract document number from filename
int extractNumber(const string &filename)
{
  string fname = fs::path(filename).filename().string();
  // Handle filenames like docExp1_X.txt or docExp2_XX.txt
  if (fname.size() >= 14) // For double digit numbers
  {
    return stoi(string(1, fname[8]) + string(1, fname[9]));
  }
  else // For single digit numbers
  {
    return stoi(string(1, fname[8]));
  }
}

// Function to determine category from directory path
string determineCategory(const string &inputDirectory)
{
  if (inputDirectory.find("real") != string::npos)
  {
    return "real";
  }
  else if (inputDirectory.find("virtual") != string::npos)
  {
    return "virtual";
  }
  return "default"; // Default fallback
}

// Function to write results to a CSV file
void writeResultsToCSV(const string &filename1,
                       const string &filename2,
                       const vector<SimilarityResult> &results)
{
  // Ensure filename has .csv extension
  string csvFilename = filename1;
  if (csvFilename.substr(csvFilename.length() - 4) != ".csv")
  {
    csvFilename += ".csv";
  }

  // Create directory if it doesn't exist
  fs::path csvPath(csvFilename);
  if (!fs::exists(csvPath.parent_path()))
  {
    fs::create_directories(csvPath.parent_path());
  }

  ofstream file(csvFilename);
  if (!file.is_open())
  {
    cerr << "Error: Unable to open file " << csvFilename << " for writing" << endl;
    return;
  }

  // Write header
  file << "Document1,Document2,Similarity,IsSimilar" << endl;

  // Write data rows
  for (const auto &result : results)
  {
    int docNum1 = extractNumber(result.file1);
    int docNum2 = extractNumber(result.file2);

    file << docNum1 << ","
         << docNum2 << ","
         << fixed << setprecision(6) << result.similarity << ","
         << (result.isSimilar ? 1 : 0)
         << endl;
  }

  file.close();

  // Open new file to store the time results
  string timeFilename = filename2;
  if (timeFilename.substr(timeFilename.length() - 4) != ".csv")
  {
    timeFilename += ".csv";
  }

  // Create directory if it doesn't exist
  fs::path timePath(timeFilename);
  if (!fs::exists(timePath.parent_path()))
  {
    fs::create_directories(timePath.parent_path());
  }

  ofstream fileTime(timeFilename);
  if (!fileTime.is_open())
  {
    cerr << "Error: Unable to open file " << timeFilename << " for writing" << endl;
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

vector<vector<int>> LSH2(const vector<int> &signature1, const int &b)
{
  // Divide Signature into b sub signatures
  vector<vector<int>> subSignatures(b);

  for (int i = 0; i < signature1.size(); i++)
  {
    subSignatures[i % b].push_back(signature1[i]);
  }
  // Hash each band
  // For each subvector 1,2,...,b
  for (int i = 0; i < b; i++)
  {
    // Apply a different hash function to each subvector
    // IMPORTANT: apply the same hash function to all subvectors 1, a different function to all subvectors 2, etc.
    // so that when comparing subvector 1 of sig1 with subvector 1 of sig2, subvector 2 of sig1 with subvector 2 of sig2, etc.
    for (int j = 0; j < subSignatures[i].size(); j++)
    {
      subSignatures[i][j] = (hashCoefficients[i].first * subSignatures[i][j] +
                             hashCoefficients[i].second) %
                            p;
    }
  }

  return subSignatures;
}

bool LSH(const vector<int> &signature1, const vector<int> &signature2,
         const int &b)
{
  // Divide Signature into b sub signatures
  vector<vector<int>> subSignatures1 = LSH2(signature1, b);
  vector<vector<int>> subSignatures2 = LSH2(signature2, b);
  // Compare each row to see if there is any i == j && [i] == [j]

  for (int i = 0; i < b; i++)
  {
    if (subSignatures1[i] == subSignatures2[i])
    {
      return true;
    }
  }

  return false;
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{


  string filename1,filename2,category;
  vector<SimilarityResult> results;
  auto startTime = chrono::high_resolution_clock::now();
  {
    Timer timerInit("Total time: ");
    
    stopwords = loadStopwords("stopwords-en.json");

    if (argc != 5)
    {
      cout << "Usage: " << argv[0] << " <directory> <k> <t> <b>" << endl;
      cout << "where:" << endl;
      cout << "  <directory> is the directory containing text files to compare" << endl;
      cout << "  <k> is the shingle size" << endl;
      cout << "  <t> is the number of hash functions" << endl;
      cout << "  <b> is the number of bands for LSH" << endl;
      return 1;
    }

    // Get directory path
    string dirPath = argv[1];

    // Get k value from command line
    k = stoi(argv[2]);
    if (k <= 0)
    {
      cerr << "Error: k must be positive" << endl;
      return 1;
    }

    // Get numHashFunctions (t) value from command line
    numHashFunctions = stoi(argv[3]);
    if (numHashFunctions <= 0)
    {
      cerr << "Error: t (number of hash functions) must be positive" << endl;
      return 1;
    }

    // Get b value from command line
    int b = stoi(argv[4]);
    if (b <= 0)
    {
      cerr << "Error: b must be positive" << endl;
      return 1;
    }

    if (b > numHashFunctions)
    {
      cerr << "Error: b (number of bands) cannot be greater than t (number of hash functions)" << endl;
      return 1;
    }

    // Initialize hash functions
    {
      Timer timerInit("Initialize hash functions");
      initializeHashFunctions();
    }

    // Vector to store all file paths
    vector<string> filePaths;

    // Check if directory exists
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath))
    {
      cerr << "Error: Directory not found or is not a directory: " << dirPath << endl;
      return 1;
    }

    // Collect all text files from the directory
    // cout << "Collecting files from directory: " << dirPath << endl;
    for (const auto &entry : fs::directory_iterator(dirPath))
    {
      if (entry.is_regular_file())
      {
        string extension = entry.path().extension().string();
        if (extension == ".txt" || extension == ".doc" || extension == ".md")
        {
          filePaths.push_back(entry.path().string());
        }
      }
    }

    if (filePaths.empty())
    {
      cerr << "Error: No valid text files found in directory." << endl;
      return 1;
    }

    // cout << "Found " << filePaths.size() << " files to compare." << endl;

    // Map to store file contents and their signatures
    map<string, pair<string, vector<int>>> fileContents;

    // Read all files and compute signatures
    {
      Timer timerProcessFiles("Read all files and compute signatures");
      for (const auto &filePath : filePaths)
      {
        // cout << "Processing file: " << filePath << endl;

        string content = readFile(filePath);
        if (content.empty())
        {
          cerr << "Warning: File is empty or could not be read: " << filePath << endl;
          continue;
        }

        unordered_set<string> kShingles;
        size_t estimatedSize = max(1UL, (unsigned long)content.length() / 10);
        kShingles.reserve(estimatedSize);

        tratar(content, kShingles);

        if (kShingles.empty())
        {
          cerr << "Warning: No k-shingles could be extracted from: " << filePath << endl;
          continue;
        }

        vector<int> signature = computeMinHashSignature(kShingles);
        fileContents[filePath] = make_pair(content, signature);
      }
    }

    // Store results
    

    // Compare all pairs of files
    int totalComparisons = 0;
    int similarFiles = 0;

    // cout << "\nComparing files..." << endl;

    {
      Timer timerInit("Compare all pairs of files Similarity + LSH");
      for (size_t i = 0; i < filePaths.size(); i++)
      {
        for (size_t j = i + 1; j < filePaths.size(); j++)
        {
          string file1 = filePaths[i];
          string file2 = filePaths[j];

          // Skip if either file couldn't be processed
          if (fileContents.find(file1) == fileContents.end() ||
              fileContents.find(file2) == fileContents.end())
          {
            continue;
          }

          totalComparisons++;

          vector<int> &signature1 = fileContents[file1].second;
          vector<int> &signature2 = fileContents[file2].second;

          float similarity = SimilaridadDeJaccard(signature1, signature2);
          bool isSimilar = LSH(signature1, signature2, b);

          if (isSimilar)
          {
            similarFiles++;
          }

          results.push_back({file1, file2, similarity, isSimilar});
        }
      }
    }
    // Get category from directory path
     category = determineCategory(dirPath);

  // Generate filenames for results
  stringstream ss;
  ss << "results/" << category << "/LSHbase/LSHbaseSimilarities_k" << k
     << "_b" << b
     << "_t" << numHashFunctions
     << ".csv";

     filename1 = ss.str();

  // Generate filename for time results
  stringstream ss2;
  ss2 << "results/" << category << "/LSHbase/LSHbaseTimes_k" << k
      << "_b" << b
      << "_t" << numHashFunctions
      << ".csv";

     filename2 = ss2.str();

  }


  writeResultsToCSV(filename1, filename2, results);

  // Write results to CSV files

  // Calculate and display total execution time
  auto endTime = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
  cout << "time: " << duration.count() << " ms" << endl;
}
