all: JaccardBruteForce MinHash LSH

JaccardBruteForce:	JaccardBruteForce.cpp
	g++ -O3 -Wall JaccardBruteForce.cpp -o JaccardBruteForce

MinHash: MinHash.cpp
	g++ -O3 -Wall MinHash.cpp -o MinHash

LSH: LSH.cpp
	g++ -O3 -Wall LSH.cpp -o LSH
