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
unsigned int k;                    // Tamaño de los k-shingles
const int numHashFunctions = 100;  // Numero de funciones hash para el minhash
vector<pair<int, int>>
    hashCoefficients;             // [a, b] for funcionhash(x) = (ax + b) % p
int p;                            // Prime number for hash functions
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

//---------------------------------------------------------------------------
// Treating Format
//---------------------------------------------------------------------------

// Quita signos de puntuacion y mayusculas
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

// Read content from file
string readFile(const string &filename) {
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

// Function to check if a string is a file path
bool isFilePath(const string &str) {
  return (str.find(".txt") != string::npos ||
          str.find(".doc") != string::npos || str.find(".md") != string::npos);
}

// Optimized function to find the next prime after n
int nextPrime(int n) {
  // Handle even numbers
  if (n % 2 == 0) n++;

  while (true) {
    bool isPrime = true;
    // Only need to check up to sqrt(n)
    int sqrtN = sqrt(n);

    // Start from 3 and check only odd numbers
    for (int i = 3; i <= sqrtN; i += 2) {
      if (n % i == 0) {
        isPrime = false;
        break;
      }
    }

    // Check special case for n = 1 or n = 2
    if (n == 1) isPrime = false;
    if (n == 2) isPrime = true;

    if (isPrime) return n;
    n += 2;  // Skip even numbers in search
  }
}

//---------------------------------------------------------------------------
// Jaccard Locality-Sensitive Hashing Algorithm
//---------------------------------------------------------------------------

// Initialize hash functions with random coefficients
void initializeHashFunctions() {
  p = nextPrime(
      10000);  // A prime number larger than maximum possible shingle ID

  // Use a time-based seed for better randomness
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  mt19937 gen(seed);
  uniform_int_distribution<> dis(1, p - 1);

  hashCoefficients.reserve(numHashFunctions);
  for (int i = 0; i < numHashFunctions; i++) {
    hashCoefficients.push_back({dis(gen), dis(gen)});  // {Random a, Random b}
  }
}

// Function to process text and extract k-shingles
void tratar(const string &texto, unordered_set<string> &kShingles) {
  queue<string> palabras;  // cola para tener las k palabras consecutivas
  string word;
  stringstream ss(texto);

  while (ss >> word) {  // leer palabra del stringstream
    // quitar singnos de puntuacion y mayusculas
    word = normalize(word);
    if (!is_stopword(word)) {
      if (word.empty()) continue;  // Skip empty words after normalization

      palabras.push(word);
      if (palabras.size() ==
          k) {  // si ya tenemos k palabras en la cola tenemos un k shingle!
        // Optimización: construir el shingle directamente con un string
        // estimado
        string shingle;
        shingle.reserve(k * 10);  // Reservar espacio aproximado

        queue<string> temp = palabras;
        for (size_t i = 0; i < k; i++) {
          shingle += temp.front();
          temp.pop();
          if (i < k - 1) shingle += " ";
        }

        kShingles.insert(shingle);
        // Quitamos la primera para avanzar (sliding window approach)
        palabras.pop();
      }
    }
  }
}

// Function to compute MinHash signatures
vector<int> computeMinHashSignature(const unordered_set<string> &kShingles) {
  vector<int> signature(numHashFunctions, INT_MAX);

  // Cache hash values to avoid recomputation
  hash<string> hasher;

  // For each shingle in the set
  for (const string &shingle : kShingles) {
    int shingleID = hasher(shingle);  // Convert shingle to a unique integer

    // Apply each hash function
    for (int i = 0; i < numHashFunctions; i++) {
      int a = hashCoefficients[i].first;
      int b = hashCoefficients[i].second;
      int hashValue =
          (1LL * a * shingleID + b) % p;  // Using 1LL to prevent overflow

      // Update signature with minimum hash value
      signature[i] = min(signature[i], hashValue);
    }
  }

  return signature;
}

float SimilaridadDeJaccard(const vector<int> &signature1,
                           const vector<int> &signature2) {
  int iguales = 0;  // Cambiado a int para optimización

  // Cuando 2 minhashes son iguales en una posicion significa que el shingle que
  // se ha usado para calcular esa posicion es el mismo en los 2 textos
  for (int i = 0; i < numHashFunctions; i++) {
    if (signature1[i] == signature2[i]) {
      iguales++;
    }
  }

  return static_cast<float>(iguales) / numHashFunctions;
}

vector<vector<int>> LSH2(const vector<int> &signature1, const int &b) {
  // Divide Signature into b sub signatures
  vector<vector<int>> subSignatures(b);

  for (int i = 0; i < signature1.size(); i++) {
    subSignatures[i % b].push_back(signature1[i]);
  }
  // Hash each band
  // Para subvector 1,2,...,b
  for (int i = 0; i < b; i++) {
    // Aplicamos a vada subvector la funcion hash distinta
    //  !!! IMPORTANTE: a todos los subvectores 1 les aplicamos la misma funcion
    //  hash a los 2 otras distinta ....
    //                   para despues al comparar el 1 de sig1 con el 1 de sig2,
    //                   el 2 de sig1 con el 2 de sig2, etc
    for (int j = 0; j < subSignatures[i].size(); j++) {
      // ENTIENDO QUE HE APLICADO BIEN EL HASH?
      subSignatures[i][j] = (hashCoefficients[i].first * subSignatures[i][j] +
                             hashCoefficients[i].second) %
                            p;
    }
  }

  return subSignatures;
}

bool LSH(const vector<int> &signature1, const vector<int> &signature2,
         const int &b) {
  // Divide Signature into b sub signatures
  vector<vector<int>> subSignatures1 = LSH2(signature1, b);
  vector<vector<int>> subSignatures2 = LSH2(signature2, b);
  // Comparamos en cada fila si hay algun momento tq i == j && [i] == [j]

  for (int i = 0; i < b; i++) {
    if (subSignatures1[i] == subSignatures2[i]) {
      return true;
    }
  }

  return false;
}
//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  stopwords = loadStopwords("stopwords-ca.json");

  if (argc != 5) {
    cout << "Usage: " << argv[0] << " <file1> <file2> <k> <b>" << endl;
    cout << "where k is the shingle size and b is the number of bands" << endl;
    return 1;
  }

  // Get k value from command line
  k = stoi(argv[3]);
  if (k <= 0) {
    cerr << "Error: k must be positive" << endl;
    return 1;
  }

  // Get b value from command line
  int b = stoi(argv[4]);
  if (b <= 0) {
    cerr << "Error: b must be positive" << endl;
    return 1;
  }

  // Read input files
  string text1 = readFile(argv[1]);
  string text2 = readFile(argv[2]);

  if (text1.empty() || text2.empty()) {
    cerr << "Error: One or both input files are empty or could not be read."
         << endl;
    return 1;
  }

  // Initialize hash functions
  initializeHashFunctions();

  // Process texts and extract k-shingles
  unordered_set<string> KT1, KT2;

  // Reserve space for estimated number of shingles
  size_t estimatedSize1 = max(1UL, (unsigned long)text1.length() / 10);
  size_t estimatedSize2 = max(1UL, (unsigned long)text2.length() / 10);
  KT1.reserve(estimatedSize1);
  KT2.reserve(estimatedSize2);

  // Extract shingles
  tratar(text1, KT1);
  tratar(text2, KT2);

  // Early exit if either set is empty
  if (KT1.empty() || KT2.empty()) {
    cout << "Error: No se pudieron extraer k-shingles de los textos. Verifica "
            "que los textos tengan al menos "
         << k << " palabras." << endl;
    return 1;
  }

  // Compute MinHash signatures
  vector<int> signature1 = computeMinHashSignature(KT1);
  vector<int> signature2 = computeMinHashSignature(KT2);

  // Calculate and output similarity
  float similarity = SimilaridadDeJaccard(signature1, signature2);
  cout << "Jaccard Similarity: " << similarity * 100 << "%" << endl;

  // Additional statistics
  // cout << "Número de k-shingles en texto 1: " << KT1.size() << endl;
  // cout << "Número de k-shingles en texto 2: " << KT2.size() << endl;

  bool similar = LSH(signature1, signature2, b);
  if (similar) {
    cout << "Los textos son similares" << endl;
  } else {
    cout << "Los textos no son similares" << endl;
  }

  return 0;
}