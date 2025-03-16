all: jaccardBruteForce jaccardMinHash jaccardLSH generateDocuments

jaccardBruteForce: jaccardBruteForce.cpp
	g++ -O3 -Wall jaccardBruteForce.cpp -o jaccardBruteForce

jaccardMinHash: jaccardMinHash.cpp
	g++ -O3 -Wall jaccardMinHash.cpp -o jaccardMinHash

jaccardLSH: jaccardLSH.cpp
	g++ -O3 -Wall jaccardLSH.cpp -o jaccardLSH

generateDocuments: generateDocuments.cpp
	g++ -O3 -Wall generateDocuments.cpp -o generateDocuments

clean:
	rm -f jaccardBruteForce jaccardMinHash jaccardLSH generateDocuments

distclean: clean
	rm -f *.o *.txt

.PHONY: all clean distclean