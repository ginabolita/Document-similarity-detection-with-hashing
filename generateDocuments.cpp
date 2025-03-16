#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>  // Incluye la biblioteca nlohmann/json
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using json = nlohmann::json;

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
void generaDocumentos(const vector<string>& permutaciones,
                      const string& baseFilename) {
  for (int i = 0; i < permutaciones.size(); ++i) {
    // Crea un nombre de archivo único para cada permutación
    string filename =
        baseFilename + "_Perm_" + to_string(i + 1) + ".txt";

    // Abre el archivo en modo de escritura
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: No se pudo crear el archivo " << filename << endl;
      continue;
    }

    // Escribe la permutación en el archivo
    file << permutaciones[i];
    file.close();

    cout << "Archivo generado: " << filename << endl;
  }
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
    cerr << "Error: No se pudo abrir el archivo JSON." << endl;
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

  generaDocumentos(permutaciones, "doc");

  return 0;
}