#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace nlohmann;

int d, D;
unordered_set<string> stopwords;  // Stopwords

//---------------------------------------------------------------------------
// Treating StopWords
// --------------------------------------------------------------------------
// Check if a word is a stopword
bool is_stopword(const string &word) {
  return stopwords.find(word) != stopwords.end();
}

// load stopwords from a file into stopword set
unordered_set<string> loadStopwords(const string &filename) {
  unordered_set<string> stopwords;
  ifstream file(filename);
  if (!file) {
    cerr << "Error opening file: " << filename << endl;
    return stopwords;
  }

  json j;
  file >> j;  // Parse JSON

  for (const auto &word : j) {
    stopwords.insert(word);
  }

  return stopwords;
}

string normalize(const string &word) {
  string result;
  result.reserve(word.length());
  for (char c : word) {
    if (isalpha(c)) {
      result += tolower(c);
    }
  }
  return result;
}

string readFile(const string &filename) {
  ifstream file(filename);
  if (!file.is_open()) {
    cerr << "Error opening file: " << filename << endl;
    return "";
  }

  string content;
  string line;
  while (getline(file, line)) {
    content += line + "\n";
  }

  return content;
}

string tratar(const string &texto, int d) {
  queue<string> palabras;
  string word, textoLimpio;
  stringstream ss(texto);

  while (ss >> word) {
    word = normalize(word);
    if (!is_stopword(word)) {
      palabras.push(word);
      if (palabras.size() == d) break;
    }
  }

  // Concatenate the words in the queue with spaces
  while (!palabras.empty()) {
    textoLimpio += palabras.front() + " ";
    palabras.pop();
  }

  // Remove the trailing space (if any)
  if (!textoLimpio.empty()) {
    textoLimpio.pop_back();
  }

  return textoLimpio;
}

void generarDocuments(const string &textoBase) {
  vector<string> paraules;
  stringstream ss(textoBase);
  string paraula;
  while (ss >> paraula) {
    paraules.push_back(paraula);
  }

  // Genera D documents
  random_device rd;
  mt19937 g(rd());
  for (int i = 1; i <= D; ++i) {
    // Barreja les paraules aleatòriament
    shuffle(paraules.begin(), paraules.end(), g);

    // Construeix el text permutat
    string textPermutat;
    for (const string &p : paraules) {
      textPermutat += p + " ";
    }

    // Elimina l'espai final
    if (!textPermutat.empty()) {
      textPermutat.pop_back();
    }

    // Guarda el text permutat en un fitxer
    string nomFitxer = "doc" + to_string(i) + ".txt";
    ofstream fitxer(nomFitxer);
    if (fitxer.is_open()) {
      fitxer << textPermutat;
      fitxer.close();
      cout << "Creat: " << nomFitxer << endl;
    } else {
      cerr << "Error en crear el fitxer: " << nomFitxer << endl;
    }
  }
}

int main(int argc, char *argv[]) {
  stopwords = loadStopwords("stopwords-en.json");

  if (argc != 4) {
    cout << "Usage: " << argv[0] << " <file> <d> <D>" << endl;
    cout << "where d is the number of words" << endl;
    cout << "where D is the number of generated documents" << endl;
    return 1;
  }

  // Get d value from command line
  d = stoi(argv[2]);
  if (d <= 0) {
    cerr << "Error: d must be positive" << endl;
    return 1;
  }

  // Get D value from command line
  D = stoi(argv[3]);
  if (D <= 0) {
    cerr << "Error: D must be positive" << endl;
    return 1;
  }

  string text = readFile(argv[1]);
  if (text.empty()) {
    cerr << "Error: Input file is empty or could not be read." << endl;
    return 1;
  }

  // me devuelve un texto con d palabras que no son stopwords
  string textoBase = tratar(text, d);

  generarDocuments(textoBase);

  return 0;
}