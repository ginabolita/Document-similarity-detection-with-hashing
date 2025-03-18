all: jaccardBruteForce jaccardMinHash jaccardLSH exp1_genRandPerm exp2_genRandShingles

flags = -O3 -Wall -std=c++17
libs = -lxxhash

jaccardBruteForce: jaccardBruteForce.cpp
	g++ ${flags} jaccardBruteForce.cpp -o jaccardBruteForce

jaccardMinHash: jaccardMinHash.cpp
	g++ ${flags} jaccardMinHash.cpp -o jaccardMinHash

jaccardLSH: jaccardLSH.cpp
	g++ ${flags} jaccardLSH.cpp -o jaccardLSH ${libs}

exp1_genRandPerm: exp1_genRandPerm.cpp
	g++ ${flags} exp1_genRandPerm.cpp -o exp1_genRandPerm

exp2_genRandShingles: exp2_genRandShingles.cpp
	g++ ${flags} exp2_genRandShingles.cpp -o exp2_genRandShingles

clean:
	rm -f jaccardBruteForce jaccardMinHash jaccardLSH exp1_genRandPerm exp2_genRandShingles

distclean: clean
	rm -f *.o *.txt

.PHONY: all clean distclean