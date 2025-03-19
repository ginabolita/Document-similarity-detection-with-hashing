#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <queue>
#include <sstream>
#include <random>
#include <climits>
#include <cctype>
#include <fstream>
#include <chrono>
#include <cmath>
#include "deps/nlohmann/json.hpp" 


using namespace std;
using namespace nlohmann;
unsigned int k;                          // Tamaño de los k-shingles
const int numHashFunctions = 100;        // Numero de funciones hash para el minhash
vector<pair<int, int>> hashCoefficients; // [a, b] for funcionhash(x) = (ax + b) % p
int p;                                   // Prime number for hash functions
unordered_set<string> stopwords;         // Stopwords


// StopWordsZone ------------------------------------------------------------------------

// Check if a word is a stopword
bool is_stopword(const string &word)
{
    return stopwords.find(word) != stopwords.end();
}

// load stopwords from a file into stopword set
unordered_set<string> loadStopwords(const string &filename)
{
    unordered_set<string> stopwords;
    ifstream file(filename);
    if (!file)
    {
        cerr << "Error opening file: " << filename << endl;
        return stopwords;
    }

    json j;
    file >> j; // Parse JSON

    for (const auto &word : j)
    {
        stopwords.insert(word.get<string>());
    }

    return stopwords;
}

//-----------------------------------------------------------------------------------------









//Format Zone ----------------------------------------------------------------------------

// Quita signos de puntuacion y mayusculas
string normalize(const string &word)
{
    string result;
    result.reserve(word.length());
    for (char c : word)
    {
        if (isalpha(c))
        {
            result += tolower(c);
        }
    }
    return result;
}

// Function to read text from a file
string readFromFile(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return "";
    }

    stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    return buffer.str();
}

// Read content from file
std::string readFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return "";
    }

    std::string content;
    std::string line;
    while (std::getline(file, line))
    {
        content += line + " ";
    }

    return content;
}

// Function to check if a string is a file path
bool isFilePath(const string &str)
{
    return (str.find(".txt") != string::npos ||
            str.find(".doc") != string::npos ||
            str.find(".md") != string::npos);
}

// Optimized function to find the next prime after n
int nextPrime(int n)
{
    // Handle even numbers
    if (n % 2 == 0)
        n++;

    while (true)
    {
        bool isPrime = true;
        // Only need to check up to sqrt(n)
        int sqrtN = sqrt(n);

        // Start from 3 and check only odd numbers
        for (int i = 3; i <= sqrtN; i += 2)
        {
            if (n % i == 0)
            {
                isPrime = false;
                break;
            }
        }

        // Check special case for n = 1 or n = 2
        if (n == 1)
            isPrime = false;
        if (n == 2)
            isPrime = true;

        if (isPrime)
            return n;
        n += 2; // Skip even numbers in search
    }
}

//----------------------------------------------------------------------------------------










// Algoritmo Zone ---------------------------------------------------------------------------

// Initialize hash functions with random coefficients
void initializeHashFunctions()
{
    p = nextPrime(10000); // A prime number larger than maximum possible shingle ID

    // Use a time-based seed for better randomness
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> dis(1, p - 1);

    hashCoefficients.reserve(numHashFunctions);
    for (int i = 0; i < numHashFunctions; i++)
    {
        hashCoefficients.push_back({dis(gen), dis(gen)}); // {Random a, Random b}
    }
}

// Function to process text and extract k-shingles
void tratar(const string &texto, unordered_set<string> &kShingles)
{
    queue<string> palabras; // cola para tener las k palabras consecutivas
    string word;
    stringstream ss(texto);

    while (ss >> word)
    { // leer palabra del stringstream
        // quitar singnos de puntuacion y mayusculas
        word = normalize(word);
        if (!is_stopword(word))
        {
            //cout << word << " ";
            if (word.empty())
                continue; // Skip empty words after normalization

            palabras.push(word);
            if (palabras.size() == k)
            { // si ya tenemos k palabras en la cola tenemos un k shingle!
                // Optimización: construir el shingle directamente con un string estimado
                string shingle;
                shingle.reserve(k * 10); // Reservar espacio aproximado

                queue<string> temp = palabras;
                for (size_t i = 0; i < k; i++)
                {
                    shingle += temp.front();
                    temp.pop();
                    if (i < k - 1)
                        shingle += " ";
                }

                kShingles.insert(shingle);
                // Quitamos la primera para avanzar (sliding window approach)
                palabras.pop();
            }
        }
    }
}

// Function to compute MinHash signatures
vector<int> computeMinHashSignature(const unordered_set<string> &kShingles)
{
    vector<int> signature(numHashFunctions, INT_MAX);

    // Cache hash values to avoid recomputation
    hash<string> hasher;

    // For each shingle in the set
    for (const string &shingle : kShingles)
    {
        int shingleID = hasher(shingle); // Convert shingle to a unique integer

        // Apply each hash function
        for (int i = 0; i < numHashFunctions; i++)
        {
            int a = hashCoefficients[i].first;
            int b = hashCoefficients[i].second;
            int hashValue = (1LL * a * shingleID + b) % p; // Using 1LL to prevent overflow

            // Update signature with minimum hash value
            signature[i] = min(signature[i], hashValue);
        }
    }

    return signature;
}

// Ehh bueno da la similaridad de Jaccard evidentemente
float SimilaridadDeJaccard(const vector<int> &signature1, const vector<int> &signature2)
{
    int iguales = 0; // Cambiado a int para optimización

    // Cuando 2 minhashes son iguales en una posicion significa que el shingle que se ha
    // usado para calcular esa posicion es el mismo en los 2 textos
    for (int i = 0; i < numHashFunctions; i++)
    {
        if (signature1[i] == signature2[i])
        {
            iguales++;
        }
    }

    return static_cast<float>(iguales) / numHashFunctions;
}

//-------------------------------------------------------------------------------------------



int main(int argc, char *argv[])
{
    stopwords = loadStopwords("stopwords-en.json");

    if (argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <file1> <file2> <k>" << std::endl;
        std::cout << "where k is the shingle size" << std::endl;
        return 1;
    }

    // Get k value from command line
    k = std::stoi(argv[3]);
    if (k <= 0)
    {
        std::cerr << "Error: k must be positive" << std::endl;
        return 1;
    }

    // Read input files
    std::string text1 = readFile(argv[1]);
    std::string text2 = readFile(argv[2]);

    if (text1.empty() || text2.empty())
    {
        std::cerr << "Error: One or both input files are empty or could not be read." << std::endl;
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
    if (KT1.empty() || KT2.empty())
    {
        cout << "Error: No se pudieron extraer k-shingles de los textos. Verifica que los textos tengan al menos " << k << " palabras." << endl;
        return 1;
    }

    // Compute MinHash signatures
    vector<int> signature1 = computeMinHashSignature(KT1);
    vector<int> signature2 = computeMinHashSignature(KT2);

    // Calculate and output similarity
    float similarity = SimilaridadDeJaccard(signature1, signature2);
    cout << "Min Hash Jaccard Similarity : " << similarity * 100 << "%" << endl;

    // Additional statistics
    //cout << "Número de k-shingles en texto 1: " << KT1.size() << endl;
    //cout << "Número de k-shingles en texto 2: " << KT2.size() << endl;

    return 0;
}
