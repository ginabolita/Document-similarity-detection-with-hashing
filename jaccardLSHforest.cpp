#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include "deps/nlohmann/json.hpp"
#include "deps/xxhash/xxhash.h"
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <functional>
#include <regex>

using namespace std;
using namespace nlohmann;
unsigned int k; // Tamaño de los k-shingles
unsigned int t; // Numero de funciones hash para el minhash (replaces numHashFunctions)
float SIMILARITY_THRESHOLD;
vector<pair<int, int>> hashCoefficients; // [a, b] for funcionhash(x) = (ax + b) % p
int p;									 // Prime number for hash functions
unordered_set<string> stopwords;		 // Stopwords
vector<pair<int, int>> similarPairs;	 // Similar pairs of documents
map<string, int> timeResults;			 // Map to store execution times

// Document structure to store document information
struct Document
{
	string filename;
	unordered_set<string> kShingles;
	vector<int> signature;

	Document(const string &name) : filename(name) {}
};

struct LSHForestNode
{
	unordered_map<int, LSHForestNode *> children;
	vector<int> docIndices;

	~LSHForestNode()
	{
		for (auto &pair : children)
		{
			delete pair.second;
		}
	}
};

// LSH Forest structure 
vector<LSHForestNode *> lshForest;

//---------------------------------------------------------------------------
// Performance Measurement <- Marcel, el timer para y mide el tiempo automaticamente cuando se destruye
//---------------------------------------------------------------------------
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
		auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
		timeResults[operationName] = duration;
	}
};

// extractNumber function to extract document number from filename
int extractNumber(const std::string &filename)
{
	std::regex pattern1(R"(docExp1_(\d+)\.txt)");
	std::regex pattern2(R"(docExp2_(\d+)\.txt)");
	std::smatch match;
	if (std::regex_search(filename, match, pattern1) or std::regex_search(filename, match, pattern2))
	{
		return std::stoi(match[1]);
	}
	return -1; // Si no se encuentra un número, devuelve -1 o maneja el error como prefieras
}

// Initialize LSH Forest 
void initializeLSHForest(int numTrees)
{
	// Clean up any existing trees
	for (auto *tree : lshForest)
	{
		delete tree;
	}

	lshForest.clear();
	lshForest.resize(numTrees, nullptr);

	// Initialize root nodes
	for (int i = 0; i < numTrees; i++)
	{
		lshForest[i] = new LSHForestNode();
	}

	// cout << "Initialized " << numTrees << " LSH Forest trees" << endl;
}

//---------------------------------------------------------------------------
// Treating StopWords
// --------------------------------------------------------------------------
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

//---------------------------------------------------------------------------
// Treating Format
//---------------------------------------------------------------------------

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

// Read content from file
string readFile(const string &filename)
{
	ifstream file(filename);
	if (!file.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		return "";
	}

	string content;
	string line;
	while (getline(file, line))
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
	if (n <= 2)
		return 2;
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

		if (isPrime)
			return n;
		n += 2; // Skip even numbers in search
	}
}

//---------------------------------------------------------------------------
// Improved Hash Functions
//---------------------------------------------------------------------------

// xxHash implementation for better hash performance
size_t xxHashFunction(const string &str, uint64_t seed = 0)
{
	return XXH64(str.c_str(), str.length(), seed);
}

// Initialize hash functions with random coefficients
void initializeHashFunctions()
{
	p = nextPrime(INT_MAX / 2); // A larger prime number for better distribution

	// Use a time-based seed for better randomness
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	mt19937_64 gen(seed); // Using 64-bit Mersenne Twister
	uniform_int_distribution<int64_t> dis(1, p - 1);

	hashCoefficients.reserve(t);
	for (unsigned int i = 0; i < t; i++)
	{
		hashCoefficients.push_back({dis(gen), dis(gen)}); // {Random a, Random b}
	}
}

//---------------------------------------------------------------------------
// Jaccard Locality-Sensitive Hashing Algorithm
//---------------------------------------------------------------------------

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
		if (!word.empty() && !is_stopword(word))
		{ // Skip empty words and stopwords
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

// Improved function to compute MinHash signatures using xxHash
vector<int> computeMinHashSignature(const unordered_set<string> &kShingles)
{
	vector<int> signature(t, INT_MAX);

	// For each shingle in the set
	for (const string &shingle : kShingles)
	{
		// Use xxHash for better distribution and performance
		uint64_t shingleID = xxHashFunction(shingle);

		// Apply each hash function
		for (unsigned int i = 0; i < t; i++)
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

// Calculate estimated Jaccard similarity using MinHash signatures
float estimatedJaccardSimilarity(const vector<int> &signature1, const vector<int> &signature2)
{
	int matchingElements = 0;

	// Count positions where signatures match
	for (size_t i = 0; i < signature1.size(); i++)
	{
		if (signature1[i] == signature2[i])
		{
			matchingElements++;
		}
	}

	return static_cast<float>(matchingElements) / signature1.size();
}

// Create a hash for a band (sub-signature)
size_t hashBand(const vector<int> &band)
{
	size_t hashValue = 0;
	for (int value : band)
	{
		// Combine hash values - similar to boost::hash_combine
		hashValue ^= hash<int>{}(value) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
	}
	return hashValue;
}

void insertIntoLSHForest(const vector<int> &signature, int docIndex, int numTrees)
{
	// Calculate prefix length for each tree
	int prefixLength = signature.size() / numTrees;
	if (prefixLength == 0)
		prefixLength = 1;

	// cout << "Adding document " << docIndex << " to LSH Forest (signature size: "
	//     << signature.size() << ", prefix length: " << prefixLength << ")" << endl;

	// For each tree in the forest
	for (int t = 0; t < numTrees; t++)
	{
		// Extract the signature prefix for this tree
		int startIdx = t * prefixLength;
		int endIdx = min((t + 1) * prefixLength, static_cast<int>(signature.size()));

		// Navigate the trie and insert document
		LSHForestNode *currentNode = lshForest[t];

		for (int i = startIdx; i < endIdx; i++)
		{
			int hashValue = signature[i];

			// Create path if it doesn't exist
			if (currentNode->children.find(hashValue) == currentNode->children.end())
			{
				currentNode->children[hashValue] = new LSHForestNode();
			}

			// Move to next node
			currentNode = currentNode->children[hashValue];

			// Add document to each node along the path
			currentNode->docIndices.push_back(docIndex);
		}
	}
}

// Depth-first search to find all document indices within a given depth
void collectDocumentIndices(LSHForestNode *node, int depth, int maxDepth, unordered_set<int> &docIndices)
{
	if (!node || depth > maxDepth)
		return;

	// Add documents at current node
	docIndices.insert(node->docIndices.begin(), node->docIndices.end());

	// Continue search through children
	for (auto &child : node->children)
	{
		collectDocumentIndices(child.second, depth + 1, maxDepth, docIndices);
	}
}

// Query the LSH Forest for similar documents (replaces findSimilarDocumentPairs)
vector<pair<int, int>> queryLSHForest(const vector<Document> &documents, int numTrees)
{
	// cout << "Starting LSH Forest query with " << documents.size()
	//      << " documents, " << numTrees << " trees, SIMILARITY_THRESHOLD " << SIMILARITY_THRESHOLD << endl;

	// Calculate maximum depth based on SIMILARITY_THRESHOLD
	// The depth corresponds to prefix length: deeper = more stringent matching
	int maxDepth = static_cast<int>((1.0 - SIMILARITY_THRESHOLD) * (documents[0].signature.size() / numTrees));

	// cout << "Using max depth of " << maxDepth << " for SIMILARITY_THRESHOLD " << SIMILARITY_THRESHOLD << endl;

	// Set to store pairs of similar documents (to avoid duplicates)
	set<pair<int, int>> similarPairsSet;

	// For each document, query the forest
	for (size_t i = 0; i < documents.size(); i++)
	{
		const Document &doc = documents[i];
		int prefixLength = doc.signature.size() / numTrees;
		if (prefixLength == 0)
			prefixLength = 1;

		// For each tree
		for (int t = 0; t < numTrees; t++)
		{
			// Calculate start index for this tree's prefix
			int startIdx = t * prefixLength;

			// Navigate the trie to find matching prefix
			LSHForestNode *currentNode = lshForest[t];
			int depth = 0;

			// Follow exact path as far as possible
			while (depth < prefixLength && currentNode)
			{
				int hashValue = doc.signature[startIdx + depth];

				if (currentNode->children.find(hashValue) != currentNode->children.end())
				{
					currentNode = currentNode->children[hashValue];
					depth++;
				}
				else
				{
					break;
				}
			}

			// Collect candidate documents at this depth and below (up to maxDepth)
			unordered_set<int> candidates;
			collectDocumentIndices(currentNode, depth, depth + maxDepth, candidates);

			// Generate document pairs
			for (int docId : candidates)
			{
				// Skip self-comparison
				if (docId == static_cast<int>(i))
					continue;

				// Ensure consistent ordering (smaller index first)
				int doc1 = static_cast<int>(i);
				int doc2 = docId;
				if (doc1 > doc2)
					swap(doc1, doc2);

				similarPairsSet.insert({doc1, doc2});
			}
		}
	}

	// Convert set to vector
	vector<pair<int, int>> similarPairs(similarPairsSet.begin(), similarPairsSet.end());

	// cout << "Found " << similarPairs.size() << " candidate pairs" << endl;

	// Filter pairs based on actual similarity
	vector<pair<int, int>> filteredPairs;
	for (const auto &pair : similarPairs)
	{
		float similarity = estimatedJaccardSimilarity(
			documents[pair.first].signature,
			documents[pair.second].signature);

		if (similarity >= SIMILARITY_THRESHOLD)
		{
			filteredPairs.push_back(pair);
			// cout << "Confirmed similar pair: " << documents[pair.first].filename
			//     << " and " << documents[pair.second].filename
			//    << " (similarity: " << similarity << ")" << endl;
		}
	}

	return filteredPairs;
}

void cleanupLSHForest()
{
	for (auto *tree : lshForest)
	{
		delete tree;
	}
	lshForest.clear();
}

// Function to write similarity results to CSV
void writeResultsToCSV(const string &filename1,
					   const string &filename2,
					   const vector<pair<int, int>> &similarPairs,
					   const vector<Document> &documents)
{
	// Ensure filename has .csv extension
	string csvFilename = filename1;
	if (csvFilename.substr(csvFilename.length() - 4) != ".csv")
	{
		csvFilename += ".csv";
	}

	ofstream file(csvFilename);
	if (!file.is_open())
	{
		cerr << "Error: Unable to open file " << csvFilename << " for writing" << endl;
		return;
	}

	// Write header
	file << "Document1,Document2,EstimatedSimilarity" << endl;

	// Write data rows
	for (const auto &pair : similarPairs)
	{
		// Extract document IDs from filenames
		string doc1 = documents[pair.first].filename;
		string doc2 = documents[pair.second].filename;

		// Parse document IDs from filenames
		string id1, id2;

		// Extract document number from filename
		int docNum1 = extractNumber(doc1);
		if (docNum1 != -1)
		{
			id1 = to_string(docNum1);
		}
		else
		{
			id1 = doc1;
		}
		int docNum2 = extractNumber(doc2);
		if (docNum2 != -1)
		{
			id2 = to_string(docNum2);
		}
		else
		{
			id2 = doc2;
		}

		// Calculate similarities
		float estSimilarity = estimatedJaccardSimilarity(
			documents[pair.first].signature,
			documents[pair.second].signature);

		// Write to CSV with fixed precision
		file << id1 << ","
			 << id2 << ","
			 << fixed << setprecision(6) << estSimilarity << ","
			 << endl;
	}

	file.close();

	// open new file to store the time results
	string timeFilename = filename2;
	if (timeFilename.substr(timeFilename.length() - 4) != ".csv")
	{
		timeFilename += ".csv";
	}

	ofstream fileTime(timeFilename);
	if (!fileTime.is_open())
	{
		cerr << "Error: Unable to open file TimeResults.csv for writing" << endl;
		return;
	}

	// Write header
	fileTime << "Operation,Time(ms)" << endl;

	for (const auto &pair : timeResults)
	{
		fileTime << pair.first << "," << pair.second << endl;
	}

	fileTime.close();
	cout << "Results written to " << csvFilename << endl;
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
void printUsage(const char *programName)
{
	cout << "1. Compare two files: " << programName << " <file1> <file2> <k> <b> <t> <sim_threshold>" << endl;
	cout << "2. Compare one file with corpus: " << programName << " <file> <corpus_dir> <k> <b> <t> <sim_threshold>" << endl;
	cout << "3. Compare all files in corpus: " << programName << " <corpus_dir> <k> <b> <t> <sim_threshold>" << endl;
	cout << "where k is the shingle size, b is the number of bands and t the number of hash functions" << endl;
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

int main(int argc, char *argv[])
{
	// Start global timer for entire execution
	auto startTime = chrono::high_resolution_clock::now();
	// Load stopwords
	stopwords = loadStopwords("stopwords-en.json");

	// Check command line arguments
	if (argc < 6 || argc > 7)
	{
		printUsage(argv[0]);
		return 1;
	}

	// Parse common parameters
	int paramOffset = 0;
	string path1, path2;
	bool corpusMode = false;
	bool singleVsCorpusMode = false;

	// Determine mode based on arguments
	if (argc == 6)
	{ // Corpus mode: program <corpus_dir> <k> <b> <t> <sim_threshold>
		path1 = argv[1];
		if (!filesystem::is_directory(path1))
		{
			cerr << "Error: " << path1 << " is not a directory" << endl;
			return 1;
		}
		corpusMode = true;
		paramOffset = 1;
	}
	else if (argc == 7)
	{ // Two files or one file vs corpus
		path1 = argv[1];
		path2 = argv[2];

		if (filesystem::is_directory(path2))
		{
			singleVsCorpusMode = true;
		}
		else if (!filesystem::is_regular_file(path1) || !filesystem::is_regular_file(path2))
		{
			cerr << "Error: One or both paths are not valid files" << endl;
			return 1;
		}
		paramOffset = 2;
	}

	// Get k value from command line
	k = stoi(argv[1 + paramOffset]);
	if (k <= 0)
	{
		cerr << "Error: k must be positive" << endl;
		return 1;
	}

	// Get b value from command line
	int b = stoi(argv[2 + paramOffset]);
	if (b <= 0)
	{
		cerr << "Error: b must be positive" << endl;
		return 1;
	}

	// Get t value from command line
	t = stoi(argv[3 + paramOffset]);
	if (t <= 0)
	{
		cerr << "Error: t must be positive" << endl;
		return 1;
	}

	// Get t value from command line
	SIMILARITY_THRESHOLD = stof(argv[4 + paramOffset]);
	cout << "Using " << b << " bands with threshold " << SIMILARITY_THRESHOLD << endl;
	if (t <= 0)
	{
		cerr << "Error: similarity threshold must be positive" << endl;
		return 1;
	}

	// Adjust number of bands based on threshold
	// cout << "Using " << b << " bands with threshold " << SIMILARITY_THRESHOLD << endl;

	// If threshold is very low, suggest using more bands
	if (SIMILARITY_THRESHOLD < 0.1 && b < 50)
	{
		cout << "Warning: For low threshold (" << SIMILARITY_THRESHOLD
			 << "), consider using more bands (current: " << b << ")" << endl;
	}

	// bloque de codigo para que al finalizar se destruya el timer (y mida el tiempo automaticamente)
	{ // Initialize hash functions
		Timer timerInit("Initialize hash functions");
		initializeHashFunctions();
	}

	// Process documents
	vector<Document> documents;

	if (corpusMode)
	{
		cout << "\nFormat: " << endl;
		cout << "doc1 | doc2 | estimated_similarity" << endl;
		// Process all files in corpus directory
		{
			Timer timerProcessCorpus("Processing corpus");
			// cout << "Processing files in directory: " << path1 << endl;

			for (const auto &entry : filesystem::directory_iterator(path1))
			{
				if (entry.is_regular_file() && isFilePath(entry.path().string()))
				{
					string filename = entry.path().string();
					Document doc(filename);

					// Read and process file
					string content = readFile(filename);
					tratar(content, doc.kShingles);

					// Compute MinHash signature
					doc.signature = computeMinHashSignature(doc.kShingles);

					documents.push_back(doc);
					// cout << "Processed: " << filename << " - " << doc.kShingles.size() << " shingles" << endl;
				}
			}
		}

		// Initialize LSH forest
		cout << "Initializing LSH Forest with " << b << " trees" << endl;
		initializeLSHForest(b);

		{
			// Add documents to LSH forest
			Timer timerLSH("LSH Forest insertion");
			for (size_t i = 0; i < documents.size(); i++)
			{
				insertIntoLSHForest(documents[i].signature, i, b);
			}
		}

		{
			// Find similar document pairs
			Timer timerFindSimilar("Finding similar documents");
			similarPairs = queryLSHForest(documents, b);
		}

		std::string category = determineCategory(argv[1]);

		// Ensure the category is valid
		if (category == "unknown") {
			std::cerr << "Warning: Could not determine category from input directory!" << std::endl;
			return 1;
		}

		// Report results
		// cout << "\nFound " << similarPairs.size() << " similar document pairs:" << endl;
		// Construct filename using a stringstream
		std::stringstream ss;
		ss << "results/" << category << "/forestSimilarities_k" << k
		   << "_t" << t
		   << "_b" << b
		   << "_threshold_" << SIMILARITY_THRESHOLD << ".csv";

		std::string filename1 = ss.str();

		// Generate the second filename with the same structure (e.g., for time measurements)
		std::stringstream ss2;
		ss2 << "results/" << category << "/forestTimes_k" << k
			<< "_t" << t
			<< "_b" << b
			<< "_threshold_" << SIMILARITY_THRESHOLD << ".csv";

		std::string filename2 = ss2.str();
		writeResultsToCSV(filename1, filename2, similarPairs, documents);
	}
	/*
	else if (singleVsCorpusMode)
	{
		// Single vs corpus mode
		Document queryDoc(path1);
		vector<pair<float, int>> similarities;

		{
			Timer timerProcessFile("Processing query file");
			cout << "Processing query file: " << path1 << endl;

			string content = readFile(path1);
			tratar(content, queryDoc.kShingles);
			queryDoc.signature = computeMinHashSignature(queryDoc.kShingles);

			cout << "Processed: " << path1 << " - " << queryDoc.kShingles.size() << " shingles" << endl;
		}

		{
			Timer timerProcessCorpus("Processing corpus");
			cout << "Processing corpus directory: " << path2 << endl;

			for (const auto &entry : filesystem::directory_iterator(path2))
			{
				if (entry.is_regular_file() && isFilePath(entry.path().string()))
				{
					string filename = entry.path().string();
					Document doc(filename);

					string content = readFile(filename);
					tratar(content, doc.kShingles);
					doc.signature = computeMinHashSignature(doc.kShingles);

					documents.push_back(doc);
					cout << "Processed: " << filename << " - " << doc.kShingles.size() << " shingles" << endl;
				}
			}
		}

		{
			Timer timerFindSimilar("Finding similar documents");

			for (size_t i = 0; i < documents.size(); i++)
			{
				float similarity = estimatedJaccardSimilarity(queryDoc.signature, documents[i].signature);
				if (similarity >= SIMILARITY_THRESHOLD)
				{
					similarities.push_back({similarity, i});
				}
			}

			// Sort by similarity (descending)
			sort(similarities.begin(), similarities.end(),
				 [](const pair<float, int> &a, const pair<float, int> &b)
				 {
					 return a.first > b.first;
				 });
		}

		// Report results
		cout << "\nFound " << similarities.size() << " similar documents to " << path1 << ":" << endl;
		for (const auto &pair : similarities)
		{
			float estSimilarity = pair.first;
			int docIndex = pair.second;

			float exactSimilarity = exactJaccardSimilarity(
				queryDoc.kShingles,
				documents[docIndex].kShingles);

			cout << "Similar document:" << endl;
			cout << "  - " << documents[docIndex].filename << endl;
			cout << "  - Estimated similarity: " << estSimilarity << endl;
			cout << "  - Exact similarity: " << exactSimilarity << endl;
			cout << endl;
		}
	}
	else
	{
		// For comparing two files
		Document doc1(path1);
		Document doc2(path2);
		float estSimilarity;
		float exactSimilarity;

		{
			Timer timerProcessFiles("Processing files");
			// cout << "Comparing two files: " << path1 << " and " << path2 << endl;

			string content1 = readFile(path1);
			string content2 = readFile(path2);

			tratar(content1, doc1.kShingles);
			tratar(content2, doc2.kShingles);

			doc1.signature = computeMinHashSignature(doc1.kShingles);
			doc2.signature = computeMinHashSignature(doc2.kShingles);

			// cout << "Processed: " << path1 << " - " << doc1.kShingles.size() << " shingles" << endl;
			// cout << "Processed: " << path2 << " - " << doc2.kShingles.size() << " shingles" << endl;
		}

		{
			Timer timerCalcSimilarity("Calculating similarity");
			estSimilarity = estimatedJaccardSimilarity(doc1.signature, doc2.signature);
			exactSimilarity = exactJaccardSimilarity(doc1.kShingles, doc2.kShingles);
		}

		// Report results
		cout << "\nSimilarity between files:" << endl;
		cout << "  - " << path1 << endl;
		cout << "  - " << path2 << endl;
		cout << "  - Estimated similarity (MinHash): " << estSimilarity << endl;
		cout << "  - Exact similarity (Jaccard): " << exactSimilarity << endl;
	}
	*/
	// Calculate and display total execution time
	auto endTime = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
	cout << "time: " << duration.count() << " ms" << endl;

	return 0;
}