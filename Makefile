all: jaccardBruteForce jaccardMinHash jaccardLSH

jaccardBruteForce: jaccardBruteForce.cpp
	g++ -O3 -Wall jaccardBruteForce.cpp -o jaccardBruteForce

jaccardMinHash: jaccardMinHash.cpp
	g++ -O3 -Wall jaccardMinHash.cpp -o jaccardMinHash

jaccardLSH: jaccardLSH.cpp
	g++ -O3 -Wall jaccardLSH.cpp -o jaccardLSH

clean:
	rm -f jaccardBruteForce jaccardMinHash jaccardLSH

distclean: clean
	rm -f *.o

.PHONY: all clean distclean