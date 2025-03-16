all: jaccardBruteForce jaccardMinHash jaccardLSH exp1_genRandPerm exp2_genRandShingles

jaccardBruteForce: jaccardBruteForce.cpp
	g++ -O3 -Wall jaccardBruteForce.cpp -o jaccardBruteForce

jaccardMinHash: jaccardMinHash.cpp
	g++ -O3 -Wall jaccardMinHash.cpp -o jaccardMinHash

jaccardLSH: jaccardLSH.cpp
	g++ -O3 -Wall jaccardLSH.cpp -o jaccardLSH

exp1_genRandPerm: exp1_genRandPerm.cpp
	g++ -O3 -Wall exp1_genRandPerm.cpp -o exp1_genRandPerm

exp2_genRandShingles: exp2_genRandShingles.cpp
	g++ -O3 -Wall exp2_genRandShingles.cpp -o exp2_genRandShingles

clean:
	rm -f jaccardBruteForce jaccardMinHash jaccardLSH exp1_genRandPerm exp2_genRandShingles

distclean: clean
	rm -f *.o *.txt

.PHONY: all clean distclean