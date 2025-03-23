#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "deps/nlohmann/json.hpp"

using namespace std;
using namespace nlohmann;
namespace fs = filesystem;

typedef unsigned int uint;
unordered_set<string> stopwords;
map<string, int> times;

struct Result
{
  string doc1;
  string doc2;
  double similarity;
};

vector<Result> results;

// Timer class to measure execution time
class Timer
{
private:
  chrono::high_resolution_clock::time_point startTime;
  string operationName;

public:
  Timer(const string &name) : operationName(name)
  {
    startTime = chrono::high_resolution_clock::now();
  }

  ~Timer()
  {
    auto endTime = chrono::high_resolution_clock::now();
    auto duration =
        chrono::duration_cast<chrono::milliseconds>(endTime - startTime)
            .count();
    if (times.count(operationName) == 0)
    {
      times[operationName] = duration;
    }
    else
    {
      times[operationName] += duration;
    }
  }
};

//---------------------------------------------------------------------------
// Load Stopwords
//---------------------------------------------------------------------------

bool is_stopword(const string &word)
{
  return stopwords.find(word) != stopwords.end();
}

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
  file >> j;

  for (const auto &word : j)
  {
    stopwords.insert(word.get<string>());
  }

  return stopwords;
}

//---------------------------------------------------------------------------
// Text Processing
//---------------------------------------------------------------------------

string remove_punctuation(const string &text)
{
  string newtext;
  for (char c : text)
  {
    if (ispunct(c))
    {
      newtext += ' ';
    }
    else
    {
      newtext += c;
    }
  }
  return newtext;
}

string normalize(const string &word)
{
  string result;
  for (char c : word)
  {
    if (isalpha(c))
    {
      result += tolower(c);
    }
  }
  return result;
}

string readFile(const string &filename)
{
  ifstream file(filename);
  if (!file.is_open())
  {
    cerr << "Error opening file: " << filename << endl;
    return "";
  }

  string content, line;
  while (getline(file, line))
  {
    content += line + " ";
  }

  return content;
}

//---------------------------------------------------------------------------
// Jaccard Brute Force Algorithm
//---------------------------------------------------------------------------

unordered_set<string> generateShingles(const string &text, uint k)
{
  unordered_set<string> shingles;
  vector<string> words;
  stringstream ss(text);
  string word;

  while (ss >> word)
  {
    string norm_word = normalize(word);
    if (!is_stopword(norm_word))
    {
      words.push_back(norm_word);
    }
  }

  if (words.size() >= k)
  {
    for (size_t i = 0; i <= words.size() - k; i++)
    {
      string shingle;
      for (size_t j = 0; j < k; j++)
      {
        if (j > 0)
          shingle += " ";
        shingle += words[i + j];
      }
      shingles.insert(shingle);
    }
  }

  return shingles;
}

double calculateJaccardSimilarity(const unordered_set<string> &set1,
                                  const unordered_set<string> &set2)
{
  int intersection = 0;
  for (const auto &shingle : set1)
  {
    if (set2.find(shingle) != set2.end())
    {
      intersection++;
    }
  }
  int unionSize = set1.size() + set2.size() - intersection;
  return unionSize > 0 ? static_cast<double>(intersection) / unionSize : 0.0;
}

//---------------------------------------------------------------------------
// Extract document number from filename
//---------------------------------------------------------------------------

string extract_doc_number(const string &filename)
{
  regex pattern(R"(docExp\d+_(\d+)\.txt)");
  smatch match;
  if (regex_search(filename, match, pattern))
  {
    return match[1].str(); // Return the extracted number
  }
  return filename; // Return full filename if no match
}

std::string determineCategory(const std::string &inputDirectory)
{
  if (inputDirectory.find("real") != std::string::npos)
  {
    return "real";
  }
  else if (inputDirectory.find("virtual") != std::string::npos)
  {
    return "virtual";
  }
  return "unknown"; // Fallback case
}

//---------------------------------------------------------------------------
// Main - Process all file pairs in a directory
//---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  int k;
  {
    Timer processTimer("time");
    if (argc != 3)
    {
      cout << "Usage: " << argv[0] << " <directory> <k>" << endl;
      return 1;
    }

    string directory = argv[1];
    k = stoi(argv[2]);
    if (k <= 0)
    {
      cerr << "Error: k must be positive" << endl;
      return 1;
    }

    stopwords = loadStopwords("stopwords-en.json");

    vector<string> files;
    for (const auto &entry : fs::directory_iterator(directory))
    {
      if (entry.path().extension() == ".txt")
      {
        files.push_back(entry.path().string());
      }
    }

    // Pre-allocate space for results
    results.reserve(files.size() * (files.size() - 1) / 2);

    for (size_t i = 0; i < files.size(); i++)
    {
      string doc1 = extract_doc_number(files[i]);
      if (doc1 == "0")
        continue; // Skip document 0

      for (size_t j = i + 1; j < files.size(); j++)
      {
        string doc2 = extract_doc_number(files[j]);
        if (doc2 == "0")
          continue; // Skip document 0

        string text1 = remove_punctuation(readFile(files[i]));
        string text2 = remove_punctuation(readFile(files[j]));

        if (text1.empty() || text2.empty())
          continue;

        double similarity = 0.0;
        unordered_set<string> shingles1;
        unordered_set<string> shingles2;

        shingles1 = generateShingles(text1, k);
        shingles2 = generateShingles(text2, k);

        similarity = calculateJaccardSimilarity(shingles1, shingles2);

        Result result;
        result.doc1 = doc1;
        result.doc2 = doc2;
        result.similarity = similarity;
        results.push_back(result);
      }
    }
  }

  // Write results to CSV
  std::string category = determineCategory(argv[1]);

  // Ensure the category is valid
  if (category == "unknown")
  {
    std::cerr << "Warning: Could not determine category from input directory!" << std::endl;
    return 1;
  }

  std::stringstream ss;
  ss << "results/" << category << "/bruteForce/bruteForceSimilarities_k" << k << ".csv";

  std::string filename1 = ss.str();

  // Generate the second filename with the same structure (e.g., for time measurements)
  std::stringstream ss2;
  ss2 << "results/" << category << "/bruteForce/bruteForceTimes_k" << k << ".csv";

  std::string filename2 = ss2.str();
 
  ofstream file(filename1);
  if (!file.is_open())
  {
    cerr << "Error: Unable to open file " << filename1 << " for writing" << endl;
    return 1;
  }

  // Write header
  file << "Doc1,Doc2,Sim%" << endl;

  for (const auto &result : results)
  {
    file << result.doc1 << "," << result.doc2 << "," << fixed << setprecision(6) << result.similarity << endl;
  }

  file.close();
  
  ofstream fileTime(filename2);
  if (!fileTime.is_open())
  {
    cerr << "Error: Unable to open file " << filename2 << " for writing" << endl;
    return 1;
  }

  // Write header
  fileTime << "Operation,Time(ms)" << endl;
  for (const auto &pair : times)
  {
    fileTime << pair.first << "," << pair.second << endl;
  }
  fileTime.close();

  return 0;
}

