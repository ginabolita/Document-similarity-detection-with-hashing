#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
using namespace std;
using namespace nlohmann;
typedef unsigned int uint;
unordered_set<string> stopwords;

//---------------------------------------------------------------------------
// Treating StopWords
// --------------------------------------------------------------------------
// Check if a word is a stopword
bool is_stopword(const string& word) {
  return stopwords.find(word) != stopwords.end();
}
// load stopwords from a file into stopword set
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
    stopwords.insert(word.get<string>()); // tenia error de tipus en el insert 
  }

  return stopwords;
}
//---------------------------------------------------------------------------
// Treating Format
//---------------------------------------------------------------------------
string remove_punctuation(string text) {
  string newtext;
  for (int i = 0; i < text.size(); ++i) {
    if (text[i] == '.' || text[i] == ',' || text[i] == '!' || text[i] == '?' ||
        text[i] == ';' || text[i] == ':') {
      newtext += ' ';
    } else {
      newtext += text[i];
    }
  }
  return newtext;
}
// Quita signos de puntuacion y mayusculas
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

// Read content from file
string readFile(const string& filename) {
  ifstream file(filename);
  if (!file.is_open()) {
    cerr << "Error opening file: " << filename << endl;
    return "";
  }

  string content;
  string line;
  while (getline(file, line)) {
    content += line + " ";
  }

  return content;
}
//---------------------------------------------------------------------------
// Jaccard Brute Force Algorithm
//---------------------------------------------------------------------------

// Function to generate k-shingles from text
unordered_set<string> generateShingles(const string& text, uint k) {
  unordered_set<string> shingles;
  vector<string> words;
  stringstream ss(text);
  string word;

  // Tokenize the text into words
  while (ss >> word) {
    // tenim en compte si es una stopword abans de tot
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

  // print shingles
  /*
  for (auto shingle : shingles){
                  cout << shingle << " ";
  }
  cout << endl;
  */

  return shingles;
}

// Calculate Jaccard similarity
double calculateJaccardSimilarity(const unordered_set<string>& set1,
                                  const unordered_set<string>& set2) {
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

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  stopwords = loadStopwords("stopwords-en.json");

  if (argc != 4) {
    cout << "Usage: " << argv[0] << " <file1> <file2> <k>" << endl;
    cout << "where k is the shingle size" << endl;
    return 1;
  }

  // Get k value from command line
  int k = stoi(argv[3]);
  if (k <= 0) {
    cerr << "Error: k must be positive" << endl;
    return 1;
  }

  // Read input files
  string text1 = readFile(argv[1]);
  string text2 = readFile(argv[2]);

  text1 = remove_punctuation(text1);
  text2 = remove_punctuation(text2);

  if (text1.empty() || text2.empty()) {
    cerr << "Error: One or both input files are empty or could not be read."
         << endl;
    return 1;
  }

  // Generate k-shingles for both documents
  unordered_set<string> shingles1 = generateShingles(text1, k);
  unordered_set<string> shingles2 = generateShingles(text2, k);

  // Calculate Jaccard similarity
  double similarity = calculateJaccardSimilarity(shingles1, shingles2);

  // Output results
  // cout << "Number of unique shingles in document 1: " << shingles1.size() <<
  // endl; cout << "Number of unique shingles in document 2: " <<
  // shingles2.size() << endl;
  cout << "Brute Force Jaccard Similarity : " << similarity * 100 << '%' << endl;

  return 0;
}
