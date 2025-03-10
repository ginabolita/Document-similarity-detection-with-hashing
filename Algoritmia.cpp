#include <iostream>
#include <vector> 
#include <set>
#include <string>
#include <queue>
#include <sstream> //para tratar los 
#include <random> //pra generar las funciones de hash
#include <climits>//se usa para inicializar el vector de minhash con INT_MAX
//se podria poner un numero muy grande en vez de INT_MAX en vrd

#include <cctype> //para las 2 funciones que se llaman en normalize (isalpha y tolower)

using namespace std;

int k;//Tamaño de los k-shingles
const int numHashFunctions = 100;  // Numero de funciones hash para el minhash
vector<pair<int, int>> hashCoefficients(numHashFunctions); //[a, b]for funcionhash(x) = (ax + b) % p

//Encuentra el siguiente numero primo despues de n
int nextPrime(int n) {
    while (true) {
        n++;
        bool isPrime = true;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) return n;
    }
}

// Initialize hash functions with random coefficients
void initializeHashFunctions() {
    int p = nextPrime(10000);  // A prime number larger than maximum possible shingle ID
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, p - 1);

    for (int i = 0; i < numHashFunctions; i++) {
        hashCoefficients[i].first = dis(gen);  // Random a
        hashCoefficients[i].second = dis(gen);  // Random b
    }
}

//Quita signos de puntuacion y mayusculas
string normalize(const string& word) {
    string result;
    for (char c : word) {
        if (isalpha(c)) {
            result += tolower(c);
        }
    }
    return result;
}

// Function to process text and extract k-shingles
void tratar(const string& texto, set<string>& kShingles) {
    queue<string> palabras;//cola para tener las k palabras consecutivas
    string word;
    stringstream ss(texto);
    while (ss >> word) { //leer palabra del stringstream
        //quitar singnos de puntuacion y mayusculas
        word = normalize(word);
        palabras.push(word);
        if (palabras.size() == k) { //si ya tenemos k palabras en la cola tenemos un k shingle!
            string shingle;
            queue<string> temp = palabras;
            while (!temp.empty()) { //vamos metiendo las palabras en el string shingle
                shingle += temp.front() + " ";
                temp.pop();
            }
            shingle.pop_back();//quitamos el espacio del final
            kShingles.insert(shingle);
            //Quitamos la primera para avanzar
            palabras.pop();
        }
    }
}

// Function to compute MinHash signatures
vector<int> computeMinHashSignature(const set<string>& kShingles) {
    vector<int> signature(numHashFunctions, INT_MAX);
    int p = nextPrime(10000);  // A prime number larger than maximum possible shingle ID

    // For each shingle in the set
    for (const string& shingle : kShingles) {
        int shingleID = hash<string>{}(shingle);  // Convert shingle to a unique integer
        
        // Apply each hash function
        for (int i = 0; i < numHashFunctions; i++) {
            int a = hashCoefficients[i].first; //esto lo inizializado antes
            int b = hashCoefficients[i].second;
            int hashValue = (a * shingleID + b) % p;
            
            // Update signature with minimum hash value
            if (hashValue < signature[i]) {
                signature[i] = hashValue;
            }
        }
    }
    
    return signature;
}

//Ehh bueno da la similaridad de Jaccard evidentemente
float SimilaridadDeJaccard(const vector<int>& signature1, const vector<int>& signature2) {
    float iguales = 0.0f; //(para luego devolver un float)
    //Cuando 2 minhashes son iguales en una posicion significa que el shingle que se ha
    // usado para calcular esa posicion es el mismo en los 2 textos
    for (int i = 0; i < numHashFunctions; i++) {
        if (signature1[i] == signature2[i]) {
            iguales++;
        }
    }
    return iguales / numHashFunctions;
}

int main() {
    cout << "Introduce el primer texto" << endl;
    string texto1;
    getline(cin, texto1); //getline como en pro1 para tratar texto con espacios

    cout << "Introduce el segundo texto" << endl;
    string texto2;
    getline(cin, texto2);

    cout << "Introduce tamaño de los k-shingles a generar" << endl;
    cin >> k;
    //'ENTENDER' ESTO Y GG
    initializeHashFunctions(); //esta en concreto es entendible pero 
    //podriamos cambiar las funciones que usa o ago para que no pareza tanto ia

    // Process texts and extract k-shingles
    set<string> KT1, KT2;
    tratar(texto1, KT1);
    tratar(texto2, KT2);

    //ENTENDER ESTO +- entedible pero hacer que no parezca tanto ia
    //en plan cambiar funciones que se usan o la estructura y sobretodo comentarlo
    // Y GG
    vector<int> signature1 = computeMinHashSignature(KT1);
    vector<int> signature2 = computeMinHashSignature(KT2);

    float similarity = SimilaridadDeJaccard(signature1, signature2);
    cout << "Jaccard Similarity: " << similarity * 100 << "%" << endl; 
    //lo he puesto en % pero lo podemos dejar en tanto por 1 si quereis
}