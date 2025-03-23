#include <algorithm>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <locale>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "deps/nlohmann/json.hpp"  // JSON library for parsing JSON files
using namespace std;
using json = nlohmann::json;

#ifdef _WIN32
#include <direct.h>  // For _mkdir on Windows
#else
#include <sys/stat.h>  // For mkdir on Unix-like systems
#endif

string delimiters =
    ".!?,;:\"'…—()[]{}/\\*";  // Common delimiters for tokenization

// Function to load stopwords from a JSON file
unordered_set<string> loadStopwords(const string& filename) {
  unordered_set<string> stopwords;
  ifstream file(filename);

  if (!file.is_open()) {
    cerr << "Warning: Could not open stopwords file: " << filename << endl;
    cerr << "Proceeding without stopwords removal" << endl;
    return stopwords;
  }

  try {
    json jsonData;
    file >> jsonData;  // Parse JSON file
    file.close();

    if (jsonData.is_array()) {
      for (const auto& word : jsonData) {
        if (word.is_string()) {
          stopwords.insert(word.get<string>());  // Add stopwords to the set
        }
      }
    } else {
      cerr << "Warning: Stopwords file is not a JSON array" << endl;
    }
  } catch (const json::exception& e) {
    cerr << "Warning: Error parsing stopwords JSON file: " << e.what() << endl;
  }

  return stopwords;
}

// Tokenization function to handle special cases like possessives, quotes, and
// em dashes
vector<string> improvedTokenization(const string& input) {
  vector<string> tokens;
  string currentToken;
  bool inQuotes = false;

  for (size_t i = 0; i < input.length(); i++) {
    char c = input[i];
    char nextChar = (i < input.length() - 1) ? input[i + 1] : '\0';

    // Handle quotes
    if (c == '"' || c == '\'' || c == '`') {
      if (!inQuotes) {
        // Opening quote - save current token if not empty
        if (!currentToken.empty()) {
          tokens.push_back(currentToken);
          currentToken.clear();
        }
        inQuotes = true;
      } else {
        // Closing quote - save current token
        inQuotes = false;
        if (!currentToken.empty()) {
          tokens.push_back(currentToken);
          currentToken.clear();
        }
      }
      continue;
    }

    // Handle possessives ('s or s')
    if (c == '\'' && (nextChar == 's' || (i > 0 && input[i - 1] == 's'))) {
      currentToken += c;  // Keep apostrophe as part of the word
      continue;
    }

    // Handle em dash (—) or double hyphen (--)
    if (c == '—' || (c == '-' && nextChar == '-')) {
      if (!currentToken.empty()) {
        tokens.push_back(currentToken);
        currentToken.clear();
      }
      if (c == '-' && nextChar == '-') {
        i++;  // Skip the second hyphen
      }
      continue;
    }

    // Handle standard punctuation
    if (!inQuotes && !isalnum(c) && c != '\'' && c != '-' && !isspace(c)) {
      if (!currentToken.empty()) {
        tokens.push_back(currentToken);
        currentToken.clear();
      }
    } else if (isspace(c)) {
      if (!currentToken.empty()) {
        tokens.push_back(currentToken);
        currentToken.clear();
      }
    } else {
      currentToken += c;  // Add alphanumeric, apostrophes, or hyphens
    }
  }

  // Add the last token if not empty
  if (!currentToken.empty()) {
    tokens.push_back(currentToken);
  }

  return tokens;
}

// Function to clean text: remove unwanted punctuation, lowercase, and remove
// stopwords
string cleanText(const string& text, const unordered_set<string>& stopwords) {
  vector<string> tokens = improvedTokenization(text);  // Tokenize text
  string spaceSeparated;
  for (const string& token : tokens) {
    spaceSeparated += token + " ";  // Join tokens with spaces
  }

  string lowercased;
  for (char c : spaceSeparated) {
    lowercased += tolower(c);  // Convert to lowercase
  }

  stringstream cleanedStream(lowercased);
  string word;
  string result;

  while (cleanedStream >> word) {
    if (stopwords.empty() || stopwords.find(word) == stopwords.end()) {
      result += word + " ";  // Remove stopwords
    }
  }

  if (!result.empty() && result.back() == ' ') {
    result.pop_back();  // Trim trailing space
  }

  return result;
}

// Function to split text into sentences
vector<string> separaFrases(const string& text,
                            const unordered_set<string>& stopwords) {
  vector<string> frases;
  string currentPhrase;

  for (size_t i = 0; i < text.length(); i++) {
    char c = text[i];
    currentPhrase += c;

    if (c == '.' || c == '!' || c == '?' || c == ',' || c == ':' || c == ';' ||
        c == '"' || c == '\'') {
      string cleanedPhrase = cleanText(currentPhrase, stopwords);
      if (!cleanedPhrase.empty()) {
        frases.push_back(cleanedPhrase);  // Add cleaned phrase
      }
      currentPhrase = "";
    }
  }

  if (!currentPhrase.empty()) {
    string cleanedPhrase = cleanText(currentPhrase, stopwords);
    if (!cleanedPhrase.empty()) {
      frases.push_back(cleanedPhrase);  // Add remaining text
    }
  }

  return frases;
}

// Function to generate random permutations of sentences
vector<string> permutabasicText(const vector<string>& frases,
                                int numpermutaciones) {
  vector<string> permutaciones;
  unsigned baseSeed = static_cast<unsigned>(time(0));

  for (int i = 0; i < numpermutaciones; ++i) {
    mt19937 generator(baseSeed + i * 1000);  // Seed for reproducibility
    vector<string> currentTemp = frases;
    shuffle(currentTemp.begin(), currentTemp.end(),
            generator);  // Shuffle sentences

    string shuffledText;
    for (const string& phrase : currentTemp) {
      shuffledText += phrase + " ";  // Build shuffled text
    }

    string normalizedText;
    bool lastWasSpace = false;
    for (char c : shuffledText) {
      if (c == ' ') {
        if (!lastWasSpace) {
          normalizedText += c;
          lastWasSpace = true;
        }
      } else {
        normalizedText += c;
        lastWasSpace = false;
      }
    }

    if (!normalizedText.empty() && normalizedText.back() == ' ') {
      normalizedText.pop_back();  // Trim trailing space
    }

    permutaciones.push_back(normalizedText);
  }

  return permutaciones;
}

// Function to generate .txt documents
void generaDocumentos(const vector<string>& permutaciones, const string& path) {
  for (int i = 0; i < permutaciones.size(); ++i) {
    string filename = path + "/docExp1_" + to_string(i + 1) + ".txt";
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Could not create file:" << filename << endl;
      continue;
    }
    file << permutaciones[i];  // Write permutation to file
    file.close();
    cout << "Generated file: " << filename << endl;
  }
}

// Function to create a directory
bool makeDirectory(const string& path) {
#ifdef _WIN32
  return _mkdir(path.c_str()) == 0;
#else
  return mkdir(path.c_str(), 0777) == 0;
#endif
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <D>" << endl;
    cout << "where D is the number of documents to generate" << endl;
    return 1;
  }

  int D = stoi(argv[1]);
  if (D <= 0) {
    cerr << "Error: D must be positive" << endl;
    return 1;
  }

  unordered_set<string> stopwords = loadStopwords("stopwords-en.json");
  cout << "Loaded " << stopwords.size() << " stopwords from file" << endl;

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
  vector<string> frases = separaFrases(basicText, stopwords);

  if (frases.empty()) {
    cerr << "Error: No valid phrases were extracted from the text" << endl;
    return 1;
  }

  vector<string> permutaciones = permutabasicText(frases, D);
  string path = "datasets/real";

  if (!makeDirectory(path)) {
    cerr << "Warning: The directory '" << path
         << "' exists or could not be created" << endl;
  } else {
    cout << "Folder created" << endl;
  }

  generaDocumentos(permutaciones, path);

  return 0;
}
