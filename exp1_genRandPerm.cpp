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

#include "deps/nlohmann/json.hpp"  // Incluye la biblioteca nlohmann/json
using namespace std;
using json = nlohmann::json;

#ifdef _WIN32
#include <direct.h>  // Para _mkdir en Windows
#else
#include <sys/stat.h>  // Para mkdir en Unix-like (Linux, macOS)
#endif

// Function to load stopwords from JSON file
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
    file >> jsonData;
    file.close();

    if (jsonData.is_array()) {
      for (const auto& word : jsonData) {
        if (word.is_string()) {
          stopwords.insert(word.get<string>());
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

//  tokenization function to handle special cases like possessives,
// quotes, and em dashes
vector<string> improvedTokenization(const string& input) {
  vector<string> tokens;
  string currentToken;
  bool inQuotes = false;

  for (size_t i = 0; i < input.length(); i++) {
    char c = input[i];
    char nextChar = (i < input.length() - 1) ? input[i + 1] : '\0';

    // Handle quotes
    if (c == '"' || c == '"' || c == '"') {
      if (!inQuotes) {
        // Opening quote - if we have a current token, save it
        if (!currentToken.empty()) {
          tokens.push_back(currentToken);
          currentToken.clear();
        }
        inQuotes = true;
      } else {
        // Closing quote - add the token with spaces
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
      // Keep the apostrophe as part of the word
      currentToken += c;
      continue;
    }

    // Handle em dash (—) or double hyphen (--)
    if (c == '—' || (c == '-' && nextChar == '-')) {
      // End current token if not empty
      if (!currentToken.empty()) {
        tokens.push_back(currentToken);
        currentToken.clear();
      }

      // Skip the second dash if it's a double hyphen
      if (c == '-' && nextChar == '-') {
        i++;
      }

      // Add a space instead
      continue;
    }

    // Handle standard punctuation
    if (!inQuotes && !isalnum(c) && c != '\'' && c != '-' && !isspace(c)) {
      // End current token if not empty
      if (!currentToken.empty()) {
        tokens.push_back(currentToken);
        currentToken.clear();
      }
      // Don't add the punctuation
    } else if (isspace(c)) {
      // Handle spaces - end current token
      if (!currentToken.empty()) {
        tokens.push_back(currentToken);
        currentToken.clear();
      }
    } else {
      // Add alphanumeric, apostrophes, hyphens within words, or text within
      // quotes
      currentToken += c;
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
  // Use improved tokenization that handles special cases
  vector<string> tokens = improvedTokenization(text);

  // Join tokens with spaces
  string spaceSeparated;
  for (const string& token : tokens) {
    spaceSeparated += token + " ";
  }

  // Convert to lowercase
  string lowercased;
  for (char c : spaceSeparated) {
    lowercased += tolower(c);
  }

  // Remove stopwords
  stringstream cleanedStream(lowercased);
  string word;
  string result;

  while (cleanedStream >> word) {
    if (stopwords.empty() || stopwords.find(word) == stopwords.end()) {
      // Keep meaningful words
      result += word + " ";
    }
  }

  // Trim trailing space if any
  if (!result.empty() && result.back() == ' ') {
    result.pop_back();
  }

  return result;
}

// Función para dividir el texto en frases
vector<string> separaFrases(const string& text,
                            const unordered_set<string>& stopwords) {
  vector<string> frases;
  stringstream ss(text);
  string phrase;

  // Divide el texto en frases usando el punto como delimitador
  while (getline(ss, phrase, '.')) {
    if (!phrase.empty()) {
      // Clean the phrase: remove punctuation, lowercase, and remove stopwords
      string cleanedPhrase = cleanText(phrase, stopwords);
      if (!cleanedPhrase.empty()) {
        frases.push_back(cleanedPhrase);
      }
    }
  }
  return frases;
}

// Función para generar permutaciones aleatorias
vector<string> permutabasicText(const vector<string>& frases,
                                int numpermutaciones) {
  vector<string> permutaciones;

  // Use consistent seed for reproducibility
  unsigned baseSeed = static_cast<unsigned>(time(0));

  for (int i = 0; i < numpermutaciones; ++i) {
    // Create a deterministic but different seed for each permutation
    mt19937 generator(baseSeed + i * 1000);

    // Create a copy for this permutation
    vector<string> currentTemp = frases;
    shuffle(currentTemp.begin(), currentTemp.end(), generator);

    // Build the shuffled text
    string shuffledText;
    for (const string& phrase : currentTemp) {
      shuffledText += phrase + " ";
    }

    // Normalize spaces (remove consecutive spaces)
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

    // Trim trailing space if any
    if (!normalizedText.empty() && normalizedText.back() == ' ') {
      normalizedText.pop_back();
    }

    permutaciones.push_back(normalizedText);
  }

  return permutaciones;
}

// Función para generar documentos .txt
void generaDocumentos(const vector<string>& permutaciones, const string& path) {
  for (int i = 0; i < permutaciones.size(); ++i) {
    // Crea un nombre de archivo único para cada permutación
    string filename = path + "/docExp1_" + to_string(i + 1) + ".txt";

    // Abre el archivo en modo de escritura
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Could not create file:" << filename << endl;
      continue;
    }

    // Escribe la permutación en el archivo
    file << permutaciones[i];
    file.close();

    cout << "Generated file: " << filename << endl;
  }
}

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

  // Load stopwords from external file
  unordered_set<string> stopwords = loadStopwords("stopwords-en.json");
  cout << "Loaded " << stopwords.size() << " stopwords from file" << endl;

  // basicText.json contiene frases
  ifstream file("basicText.json");
  if (!file.is_open()) {
    cerr << "Error: Could not open JSON file." << endl;
    return 1;
  }

  // parse json
  json jsonData;
  file >> jsonData;
  file.close();

  if (jsonData["experimento_1"]["basicText"].is_null()) {
    cerr << "Error: basicText is null in the JSON file." << endl;
    return 1;
  }

  string basicText = jsonData["experimento_1"]["basicText"];
  vector<string> frases = separaFrases(basicText, stopwords);

  // If no valid phrases were extracted, exit with an error
  if (frases.empty()) {
    cerr << "Error: No valid phrases were extracted from the text" << endl;
    return 1;
  }

  vector<string> permutaciones = permutabasicText(frases, D);
  string path = "datasets/real";

  // Crear la carpeta
  if (!makeDirectory(path)) {
    cerr << "Warning: The directory '" << path
         << "' exists or could not be created" << endl;
  } else {
    cout << "Folder created" << endl;
  }

  generaDocumentos(permutaciones, path);

  return 0;
}