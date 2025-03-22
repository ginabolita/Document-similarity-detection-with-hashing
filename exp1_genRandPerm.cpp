#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "deps/nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

unordered_set<string> stopwords;

bool is_stopword(const string& word) {
  string lowerWord = word;
  transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
  return stopwords.find(lowerWord) != stopwords.end();
}

unordered_set<string> loadStopwords(const string& filename) {
  unordered_set<string> stopwords;
  ifstream file(filename);
  if (!file) {
    cerr << "Error opening file: " << filename << endl;
    return stopwords;
  }

  json j;
  file >> j;

  for (const auto& word : j) {
    stopwords.insert(word.get<string>());
  }

  return stopwords;
}

vector<string> tokenizeWords(const string& text) {
  vector<string> words;
  stringstream ss(text);
  string word;

  while (ss >> word) {
    if (!word.empty() && ispunct(word.back())) {
      word.pop_back();
    }

    string lowerWord = word;
    transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);

    if (!lowerWord.empty() && !is_stopword(lowerWord)) {
      words.push_back(lowerWord);
    }
  }
  return words;
}

int countUniqueWords(const vector<string>& words) {
  unordered_set<string> uniqueWords(words.begin(), words.end());
  return uniqueWords.size();
}

vector<string> swapRandomPairs(const vector<string>& words, int numSwaps) {
  vector<string> modifiedWords = words;
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<size_t> dist(0, words.size() - 1);

  for (int i = 0; i < numSwaps; ++i) {
    size_t idx1 = dist(gen);
    size_t idx2 = dist(gen);

    if (idx1 != idx2) {
      swap(modifiedWords[idx1], modifiedWords[idx2]);
    }
  }

  return modifiedWords;
}

void generaDocumentos(const vector<string>& permutaciones, const string& path) {
  for (size_t i = 0; i < permutaciones.size(); ++i) {
    string filename = path + "/docExp1_" + to_string(i + 1) + ".txt";
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Could not create file:" << filename << endl;
      continue;
    }

    file << permutaciones[i];
    file.close();
    cout << "Generated file: " << filename << endl;
  }
}

bool makeDirectory(const std::string& path) {
  if (std::filesystem::exists(path)) {
    std::cout << path << std::endl;
    return true;
  }
  if (std::filesystem::create_directory(path)) {
    std::cout << path << std::endl;
    return true;
  }
  std::cerr << "Warning: The directory '" << path << "' could not be created" << std::endl;
  return false;
}

int main(int argc, char* argv[]) {
  stopwords = loadStopwords("stopwords-en.json");
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <D>" << endl;
    return 1;
  }
  int D = stoi(argv[1]);
  if (D < 20) {
    cerr << "Error: D must be at least 20 according to requirements" << endl;
    return 1;
  }

  ifstream file("basicText.json");
  if (!file.is_open()) {
    cerr << "Error: Could not open JSON file." << endl;
    return 1;
  }

  json jsonData;
  file >> jsonData;
  file.close();

  if (jsonData["experimento_1"]["basicText"].is_null()) {
    cerr << "Error: basicText is null in the JSON file." << endl;
    return 1;
  }

  string basicText = jsonData["experimento_1"]["basicText"];
  vector<string> words = tokenizeWords(basicText);

  int uniqueWordCount = countUniqueWords(words);
  if (uniqueWordCount < 50) {
    cerr << "Error: Base text must contain at least 50 different words. Current count: " << uniqueWordCount << endl;
    return 1;
  }

  cout << "Base text contains " << words.size() << " total words and " << uniqueWordCount << " unique words." << endl;

  vector<string> permutaciones;
  for (int i = 0; i < D; ++i) {
    vector<string> modifiedWords = swapRandomPairs(words, words.size() / 10);
    stringstream shuffledText;
    for (const auto& word : modifiedWords) {
      shuffledText << word << " ";
    }
    permutaciones.push_back(shuffledText.str());
  }

  string path = "datasets/real/";
  makeDirectory(path);
  generaDocumentos(permutaciones, path);

  return 0;
}
