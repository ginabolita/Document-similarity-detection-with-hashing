#include <algorithm>
#include <fstream>
#include <filesystem>
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

#ifdef _WIN32
#include <direct.h>  // Para _mkdir en Windows
#else
#include <sys/stat.h>  // Para mkdir en Unix-like (Linux, macOS)
#endif

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
    stopwords.insert(word.get<string>());
  }

  return stopwords;
}
//---------------------------------------------------------------------------
// Treating Format
//---------------------------------------------------------------------------
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

unordered_set<string> generateShingles(const string& text) {
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
  int count = 0;
  for (auto shingle : shingles) {
    ++count;
  }
  //cout << "hay :" << count << " k-shingles" << endl;

  return shingles;
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

void generaDocumentos(const unordered_set<string>& shingles, const string& path) {
  for (int i = 0; i < D; ++i) {
    // Crea un nombre de archivo único para cada permutación
    string filename = path + "/docExp2_" + to_string(i + 1) + ".txt";

    // Abre el archivo en modo de escritura
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Could not create file:" << filename << endl;
      continue;
    }

    // rango variable pero decido uno equilibrado 20-80 % por documento
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(20, 80);
    int randQuantity = dis(gen);
    vector<string> selectedShingles = selectQuantity(shingles, randQuantity);

    // Escribir los shingles seleccionados en el archivo
    for (const string& shingle : selectedShingles) {
      file << shingle << endl;
    }
    file.close();
    //cout << "Generated file: " << filename << endl;
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
    
    std::cerr << "Warning: The directory '" << path << "' could not be created" << std::endl;
    return false;
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
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
  if (D <= 0) {
    cerr << "Error: D must be positive" << endl;
    return 1;
  }

ifstream file("basicText.json");
  if (!file.is_open()) {
    cerr << "Error: Could not open JSON file." << endl;
    return 1;
  }
  // parse json
  json jsonData;
  file >> jsonData;
  file.close();
  if (jsonData["experimento_2"]["basicText"].is_null()) {
    cerr << "Error: basicText is null in the JSON file." << endl;
    return 1;
  }

  string basicText = jsonData["experimento_2"]["basicText"];

  unordered_set<string> shingles = generateShingles(basicText);
  string path = "datasets/virtual/";
      // Crear la carpeta
  makeDirectory(path);

  generaDocumentos(shingles, path);

  return 0;
}