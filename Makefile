all: JaccardBruteForce Algoritmia

JaccardBruteForce:	JaccardBruteForce.cpp
	g++ -O3 -Wall JaccardBruteForce.cpp -o JaccardBruteForce

Algoritmia: Algoritmia.cpp
	g++ -O3 -Wall Algoritmia.cpp -o Algoritmia