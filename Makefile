all: jaccardBruteForce jaccardMinHash jaccardLSH genBaseDocuments

jaccardBruteForce: jaccardBruteForce.cpp
	g++ -O3 -Wall jaccardBruteForce.cpp -o jaccardBruteForce

jaccardMinHash: jaccardMinHash.cpp
	g++ -O3 -Wall jaccardMinHash.cpp -o jaccardMinHash

jaccardLSH: jaccardLSH.cpp
	g++ -O3 -Wall jaccardLSH.cpp -o jaccardLSH

genBaseDocuments: genBaseDocuments.cpp
	g++ -O3 -Wall genBaseDocuments.cpp -o genBaseDocuments

clean:
	rm -f jaccardBruteForce  jaccardMinHash jaccardLSH genBaseDocuments

distclean: clean
	rm -f *.o *.txt

.PHONY: all clean distclean