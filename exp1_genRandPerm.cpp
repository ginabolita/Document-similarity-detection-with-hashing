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

unsigned int k, D;
unordered_set<string> stopwords;

bool is_stopword(const string& word) {
  return stopwords.find(word) != stopwords.end();
} 

// bool is_stopword(const string& word) {
//   string lowerWord = word;
//   transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
//   return stopwords.find(lowerWord) != stopwords.end();
// }

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

// Function to tokenize text into words
vector<string> tokenizeWords(const string& text) {
  vector<string> words;
  string cleanedText;

  // First pass: convert to lowercase and remove punctuation
  for (char c : text) {
    if (ispunct(c)) {
      cleanedText += ' ';  // Replace punctuation with space
    } else {
      cleanedText += tolower(c);  // Convert to lowercase
    }
  }

  // Second pass: tokenize and filter stopwords
  stringstream ss(cleanedText);
  string word;

  while (ss >> word) {
    // Skip empty words and stopwords
    if (!word.empty() && !is_stopword(word)) {
      words.push_back(word);
    }
  }

  return words;
}

// Function to count unique words in text
/* int countUniqueWords(const vector<string>& words) {
  unordered_set<string> uniqueWords;

  for (const string& word : words) {
    // Convert to lowercase for counting unique words
    string lowerWord = word;
    transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
    uniqueWords.insert(lowerWord);
  }

  return uniqueWords.size();
}
 */

// Function to count unique words in text (excluding stopwords)
int countUniqueWords(const vector<string>& words) {
  unordered_set<string> uniqueWords;

  for (const string& word : words) {
    uniqueWords.insert(word);  // Words are already lowercase and filtered
  }

  return uniqueWords.size();
}

// Function to generate random permutations of words
vector<string> permutabasicText(const vector<string>& words,
                                int numpermutaciones) {
  vector<string> permutaciones;

  // Create a proper random generator
  random_device rd;
  mt19937 gen(rd());

  for (int i = 0; i < numpermutaciones; ++i) {
    vector<string> temp = words;  // Create a copy for each permutation
    shuffle(temp.begin(), temp.end(), gen);  // Shuffle the words

    // Combine shuffled words into a document
    stringstream shuffledText;
    for (size_t j = 0; j < temp.size(); ++j) {
      shuffledText << temp[j];

      // Add space between words, and occasionally add periods to create
      // sentences
      if (j < temp.size() - 1) {
        shuffledText << " ";
        // Randomly add periods to create sentence structure (approx every 8-12
        // words)
        if (rand() % 10 == 0) {
          shuffledText << ". ";
        }
      }
    }

    // Ensure the document ends with a period
    if (shuffledText.str().back() != '.') {
      shuffledText << ".";
    }

    permutaciones.push_back(shuffledText.str());
  }

  return permutaciones;
}

// Function to generate .txt documents
void generaDocumentos(const vector<string>& permutaciones, const string& path) {
  for (int i = 0; i < permutaciones.size(); ++i) {
    // Create a unique filename for each permutation
    string filename = path + "/docExp1_" + to_string(i + 1) + ".txt";

    // Open file in write mode
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Could not create file:" << filename << endl;
      continue;
    }

    // Write the permutation to the file
    file << permutaciones[i];
    file.close();

    cout << "Generated file: " << filename << endl;
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
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <D>" << endl;
    cout << "where D is the number of documents to generate" << endl;
    return 1;
  }
  D = stoi(argv[1]);
  if (D < 20) {
    cerr << "Error: D must be at least 20 according to requirements" << endl;
    return 1;
  }

  // Open and parse JSON file
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

  // Tokenize the text into words
  vector<string> words = tokenizeWords(basicText);

  // Check if base text has at least 50 different words
  int uniqueWordCount = countUniqueWords(words);
  if (uniqueWordCount < 50) {
    cerr << "Error: Base text must contain at least 50 different words. "
            "Current count: "
         << uniqueWordCount << endl;
    return 1;
  }

  cout << "Base text contains " << words.size() << " total words and "
       << uniqueWordCount << " unique words." << endl;

  // Generate permutations at word level
  vector<string> permutaciones = permutabasicText(words, D);

  string path = "datasets/real/";
  // Create folder
  makeDirectory(path);

  generaDocumentos(permutaciones, path);

  return 0;
}