#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "deps/nlohmann/json.hpp"
using namespace std;
using namespace nlohmann;
typedef unsigned int uint;
unordered_set<string> stopwords;

unsigned int k, D;

// Treating StopWords
bool is_stopword(const string& word) {
  return stopwords.find(word) != stopwords.end();
}

unordered_set<string> loadStopwords(const string& filename) {
  unordered_set<string> stopwords;
  ifstream file(filename);
  if (!file) {
    cerr << "Error opening file: " << filename << endl;
    return stopwords;
  }

  json j;
  file >> j;  // Parse JSON

  for (const auto& word : j) {
    stopwords.insert(word.get<string>());
  }

  return stopwords;
}

// Treating Format
string normalize(const string& word) {
  string result;
  result.reserve(word.length());
  for (char c : word) {
    if (isalpha(c)) {
      result += tolower(c);
    }
  }
  return result;
}

// Count unique words in text (excluding stopwords)
int countUniqueWords(const string& text) {
  unordered_set<string> uniqueWords;
  stringstream ss(text);
  string word;

  while (ss >> word) {
    string normalizedWord = normalize(word);
    if (!normalizedWord.empty() && !is_stopword(normalizedWord)) {
      uniqueWords.insert(normalizedWord);
    }
  }

  return uniqueWords.size();
}

unordered_set<string> generateShingles(const string& text) {
  unordered_set<string> shingles;
  vector<string> words;
  stringstream ss(text);
  string word;

  // Tokenize the text into words
  while (ss >> word) {
    // Consider if it's a stopword
    if (!is_stopword(normalize(word))) {
      words.push_back(normalize(word));
    }
  }

  // Generate k-word shingles
  if (words.size() >= k) {
    for (size_t i = 0; i <= words.size() - k; i++) {
      string shingle;
      for (size_t j = 0; j < k; j++) {
        if (j > 0) shingle += " ";  // Separate words with a space
        shingle += words[i + j];
      }
      shingles.insert(shingle);
    }
  }

  cout << "Total k-shingles generated: " << shingles.size() << endl;
  return shingles;
}

// Calculate Jaccard similarity between two sets
double calculateJaccardSimilarity(const vector<string>& set1,
                                  const vector<string>& set2) {
  // Count intersection
  int intersection = 0;
  for (const auto& item : set1) {
    if (find(set2.begin(), set2.end(), item) != set2.end()) {
      intersection++;
    }
  }

  // Union size = size of set1 + size of set2 - intersection
  int unionSize = set1.size() + set2.size() - intersection;

  // Return Jaccard similarity
  return unionSize > 0 ? static_cast<double>(intersection) / unionSize : 0.0;
}

// Create a shared pool of shingles that will be used across documents
vector<string> createSharedPool(const unordered_set<string>& allShingles,
                                int sharedPoolSize) {
  vector<string> shinglesVec(allShingles.begin(), allShingles.end());
  random_device rd;
  mt19937 gen(rd());
  shuffle(shinglesVec.begin(), shinglesVec.end(), gen);

  // Take the first sharedPoolSize elements or all if less
  int actualSize = min(sharedPoolSize, static_cast<int>(shinglesVec.size()));
  return vector<string>(shinglesVec.begin(), shinglesVec.begin() + actualSize);
}

// Select shingles for a document with controlled randomness
vector<string> selectShinglesWithSimilarity(
    const vector<string>& sharedPool, const unordered_set<string>& allShingles,
    int quantity, double sharedRatio) {
  random_device rd;
  mt19937 gen(rd());

  // Calculate how many shingles to take from shared pool
  int sharedCount = static_cast<int>(quantity * sharedRatio);
  sharedCount = min(sharedCount, static_cast<int>(sharedPool.size()));

  // Select random shingles from shared pool
  vector<string> selectedShingles;
  vector<string> sharedPoolCopy = sharedPool;  // Make a copy to shuffle
  shuffle(sharedPoolCopy.begin(), sharedPoolCopy.end(), gen);
  selectedShingles.insert(selectedShingles.end(), sharedPoolCopy.begin(),
                          sharedPoolCopy.begin() + sharedCount);

  // Fill the rest with random shingles from the remaining pool
  int remainingCount = quantity - sharedCount;
  if (remainingCount > 0) {
    // Create a vector of all shingles excluding those already selected
    vector<string> remainingShingles;
    for (const auto& shingle : allShingles) {
      if (find(selectedShingles.begin(), selectedShingles.end(), shingle) ==
          selectedShingles.end()) {
        remainingShingles.push_back(shingle);
      }
    }

    // Shuffle and select from remaining shingles
    shuffle(remainingShingles.begin(), remainingShingles.end(), gen);
    int actualRemainingCount =
        min(remainingCount, static_cast<int>(remainingShingles.size()));
    selectedShingles.insert(selectedShingles.end(), remainingShingles.begin(),
                            remainingShingles.begin() + actualRemainingCount);
  }

  return selectedShingles;
}

void generaDocumentos(const unordered_set<string>& shingles,
                      const string& path) {
  random_device rd;
  mt19937 gen(rd());
  int totalShingles = shingles.size();

  // Create a reasonably sized shared pool (e.g., 60% of all shingles)
  int sharedPoolSize = totalShingles * 0.85;
  vector<string> sharedPool = createSharedPool(shingles, sharedPoolSize);
  cout << "Created shared pool with " << sharedPool.size() << " shingles"
       << endl;

  // Define parameters for document generation
  int minShingles = max(50, totalShingles / 5);
  int maxShingles = min(totalShingles * 9 / 10, totalShingles - 1);
  uniform_int_distribution<> dis(minShingles, maxShingles);

  // Set the shared ratio (how much of each document comes from shared pool)
  double sharedRatio = 0.7;  // 70% of shingles will be from shared pool

  // Store generated documents for similarity calculation
  vector<vector<string>> documents;

  for (int i = 0; i < D; ++i) {
    // Create a unique filename for each document
    string filename = path + "/docExp2_" + to_string(i + 1) + ".txt";

    // Determine how many shingles this document will have
    int docShingleCount = dis(gen);

    // Select shingles with controlled similarity
    vector<string> selectedShingles = selectShinglesWithSimilarity(
        sharedPool, shingles, docShingleCount, sharedRatio);

    // Store for similarity calculation
    documents.push_back(selectedShingles);

    // Write to file
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Could not create file:" << filename << endl;
      continue;
    }

    for (const string& shingle : selectedShingles) {
      file << shingle << endl;
    }
    file.close();
    cout << "Generated file: " << filename << " with "
         << selectedShingles.size() << " shingles" << endl;
  }

  // Generate similarity matrix using Jaccard similarity
  ofstream simMatrix("datasets/similarity_matrix.txt");
  if (simMatrix.is_open()) {
    simMatrix << "Jaccard Similarity Matrix between documents:" << endl;
    simMatrix << "Total k-shingles in base set: " << totalShingles << endl
              << endl;

    // Create a table header
    simMatrix << "Doc\t";
    for (int i = 0; i < D; ++i) {
      simMatrix << "Doc" << i + 1 << "\t";
    }
    simMatrix << endl;

    // Calculate and write actual Jaccard similarities
    for (int i = 0; i < D; ++i) {
      simMatrix << "Doc" << i + 1 << "\t";
      for (int j = 0; j < D; ++j) {
        if (i == j) {
          simMatrix << "1.000\t";  // Self-similarity is 1
        } else {
          double similarity =
              calculateJaccardSimilarity(documents[i], documents[j]);
          simMatrix << fixed << setprecision(3) << similarity << "\t";
        }
      }
      simMatrix << endl;
    }

    // Calculate and report average similarity
    double totalSimilarity = 0.0;
    int pairCount = 0;
    for (int i = 0; i < D; ++i) {
      for (int j = i + 1; j < D; ++j) {
        totalSimilarity +=
            calculateJaccardSimilarity(documents[i], documents[j]);
        pairCount++;
      }
    }
    double averageSimilarity =
        pairCount > 0 ? totalSimilarity / pairCount : 0.0;
    simMatrix << endl
              << "Average Jaccard similarity: " << fixed << setprecision(3)
              << averageSimilarity << endl;

    simMatrix.close();
    cout << "Generated similarity matrix with average similarity: "
         << averageSimilarity << endl;
  }
}

bool makeDirectory(const std::string& path) {
  if (std::filesystem::exists(path)) {
    std::cout << path << std::endl;  // Path already exists
    return true;
  }

  if (std::filesystem::create_directory(path)) {
    std::cout << path << std::endl;  // Successfully created
    return true;
  }

  std::cerr << "Warning: The directory '" << path << "' could not be created"
            << std::endl;
  return false;
}

int main(int argc, char* argv[]) {
  stopwords = loadStopwords("stopwords-en.json");
  if (argc != 3) {
    cout << "Usage: " << argv[0] << " <k> <D>" << endl;
    cout << "where k is the shingle size" << endl;
    cout << "where D is the number of documents to generate" << endl;
    return 1;
  }

  // Get k value from command line
  k = stoi(argv[1]);
  if (k <= 0) {
    cerr << "Error: k must be positive" << endl;
    return 1;
  }

  D = stoi(argv[2]);
  if (D < 20) {
    cerr << "Error: D must be at least 20 according to requirements" << endl;
    return 1;
  }

  ifstream file("basicText.json");
  if (!file.is_open()) {
    cerr << "Error: Could not open JSON file." << endl;
    return 1;
  }

  // Parse JSON
  json jsonData;
  file >> jsonData;
  file.close();

  if (jsonData["experimento_2"]["basicText"].is_null()) {
    cerr << "Error: basicText is null in the JSON file." << endl;
    return 1;
  }

  string basicText = jsonData["experimento_2"]["basicText"];

  // Check if base text has at least 100 different words
  int uniqueWordCount = countUniqueWords(basicText);
  if (uniqueWordCount < 100) {
    cerr << "Error: Base text for experiment 2 must contain at least 100 "
            "different words. Current count: "
         << uniqueWordCount << endl;
    return 1;
  }

  cout << "Base text contains " << uniqueWordCount << " unique words." << endl;

  unordered_set<string> shingles = generateShingles(basicText);
  string path = "datasets/virtual/";
  // Create folder
  makeDirectory(path);

  generaDocumentos(shingles, path);

  return 0;
}
