#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include "deps/nlohmann/json.hpp"  // Incluye la biblioteca nlohmann/json
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using json = nlohmann::json;

#ifdef _WIN32
#include <direct.h>  // Para _mkdir en Windows
#else
#include <sys/stat.h>  // Para mkdir en Unix-like (Linux, macOS)
#endif

// Función para dividir el texto en frases (se puede hacer por parrafos, pero
// como nos piden 20 documentos, con 5 frasees ya se generan las permutaciones
// necesarias)
vector<string> separaFrases(const string& text) {
  vector<string> frases;
  stringstream ss(text);
  string phrase;

  // Divide el texto en frases usando el punto como delimitador
  while (getline(ss, phrase, '.')) {
    if (!phrase.empty()) {
      frases.push_back(phrase + ".");
    }
  }
  return frases;
}

// Función para generar permutaciones aleatorias
vector<string> permutabasicText(const vector<string>& frases,
                                int numpermutaciones) {
  vector<string> permutaciones;
  vector<string> temp = frases;

  for (int i = 0; i < numpermutaciones; ++i) {
    shuffle(temp.begin(), temp.end(),
            mt19937(time(0) + i));  // Mezcla con una semilla diferente
    string shuffledText;
    for (const string& phrase : temp) {
      shuffledText += phrase + " ";
    }
    permutaciones.push_back(shuffledText);
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
  // basicText.json contiene 5 frases. Es decir 5! permutaciones en base a
  // frases posibles.
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

  vector<string> frases = separaFrases(basicText);

  vector<string> permutaciones = permutabasicText(frases, D);

  string path = "exp1_directory";
      // Crear la carpeta
  if (!makeDirectory(path)) {
    cerr << "Warning: The directory '" << path << "' exists or could not be created" << endl;
  } else {
    cout << "Folder created" << endl;
  }

  generaDocumentos(permutaciones, path);

  return 0;
}