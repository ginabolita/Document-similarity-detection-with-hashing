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

// Calculate expected similarity based on formula
// TODO: esto hace falta?
double calculateExpectedSimilarity(int ni, int nj, int n) {
  double pi = static_cast<double>(ni) / n;
  double pj = static_cast<double>(nj) / n;
  return (pi * pj) / (pi + pj - pi * pj);
}

vector<string> selectQuantity(const unordered_set<string>& shingles,
                              int quantity) {
  vector<string> selectedShingles;
  vector<string> shinglesVec(shingles.begin(), shingles.end());
  random_device rd;
  mt19937 gen(rd());
  shuffle(shinglesVec.begin(), shinglesVec.end(), gen);

  selectedShingles.reserve(quantity);
  for (int i = 0; i < quantity && i < shinglesVec.size(); ++i) {
    selectedShingles.push_back(shinglesVec[i]);
  }

  return selectedShingles;
}

void generaDocumentos(const unordered_set<string>& shingles,
                      const string& path) {
  vector<int> shingleCounts;  // Store counts for similarity calculation
  int totalShingles = shingles.size();
  random_device rd;
  mt19937 gen(rd());

  // Define a reasonable range based on total shingles available
  int minShingles = max(10, totalShingles / 10);  // At least 10 or 10% of total
  int maxShingles =
      min(totalShingles * 8 / 10, totalShingles - 1);  // At most 80% of total
  uniform_int_distribution<> dis(minShingles, maxShingles);

  for (int i = 0; i < D; ++i) {
    // Create a unique filename for each document
    string filename = path + "/docExp2_" + to_string(i + 1) + ".txt";

    // Open file in write mode
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Could not create file:" << filename << endl;
      continue;
    }

    // Random quantity between minShingles and maxShingles
    int randQuantity = dis(gen);
    vector<string> selectedShingles = selectQuantity(shingles, randQuantity);
    shingleCounts.push_back(randQuantity);

    // Write selected shingles to file
    for (const string& shingle : selectedShingles) {
      file << shingle << endl;
    }
    file.close();
    cout << "Generated file: " << filename << " with " << randQuantity
         << " shingles" << endl;
  }

  // Generate similarity matrix report
  ofstream simMatrix("datasets/similarity_matrix.txt");
  if (simMatrix.is_open()) {
    simMatrix << "Expected Similarity Matrix between documents:" << endl;
    simMatrix << "Total k-shingles in base set: " << totalShingles << endl
              << endl;

    // Create a table header
    simMatrix << "Doc\t";
    for (int i = 0; i < D; ++i) {
      simMatrix << "Doc" << i + 1 << "\t";
    }
    simMatrix << endl;

    // Create similarity matrix
    for (int i = 0; i < D; ++i) {
      simMatrix << "Doc" << i + 1 << "\t";
      for (int j = 0; j < D; ++j) {
        if (i == j) {
          simMatrix << "1.000\t";  // Self-similarity is 1
        } else {
          double similarity = calculateExpectedSimilarity(
              shingleCounts[i], shingleCounts[j], totalShingles);
          simMatrix << fixed << setprecision(3) << similarity << "\t";
        }
      }
      simMatrix << endl;
    }
    simMatrix.close();
    cout << "Generated similarity matrix" << endl;
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