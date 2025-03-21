# Detect a modern compiler (use g++-13 if available, otherwise fallback to g++)
CXX := $(shell command -v g++-13 >/dev/null 2>&1 && echo g++-13 || echo g++)
FLAGS = -O3 -Wall -std=c++17
INCLUDE = -Inlohmann -Ixxhash
LIBS = deps/xxhash/libxxhash.a

# Define output directory
OUTDIR = executables

# Automatically find all .cpp files and generate targets in the output directory
SOURCES = $(wildcard *.cpp)
EXECUTABLES = $(patsubst %.cpp,$(OUTDIR)/%, $(SOURCES))

# Default target: check dependencies, build xxhash, and compile everything
all: check-dependencies deps/xxhash/libxxhash.a | $(OUTDIR) $(EXECUTABLES)

# Create the output directory if it does not exist
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generic rule to compile any .cpp file into an executable
$(OUTDIR)/%: %.cpp | $(OUTDIR)
	$(CXX) $(FLAGS) $< $(INCLUDE) $(LIBS) -o $@

# Rule to build xxHash if it's missing
deps/xxhash/libxxhash.a: deps/xxhash/xxhash.c deps/xxhash/xxhash.h
	@echo "Building xxhash static library..."
	$(CXX) -c -fPIC deps/xxhash/xxhash.c -o deps/xxhash/xxhash.o
	ar rcs deps/xxhash/libxxhash.a deps/xxhash/xxhash.o

# Check for required dependencies
check-dependencies:
	@echo "Checking dependencies..."
	@command -v $(CXX) >/dev/null 2>&1 || { echo "Error: $(CXX) not found! Run './setup.sh' to install dependencies."; exit 1; }
	@ls deps/nlohmann/json.hpp >/dev/null 2>&1 || { echo "Error: deps/nlohmann/json.hpp not found! Run './setup.sh'."; exit 1; }
	@ls deps/xxhash/xxhash.h >/dev/null 2>&1 || { echo "Error: xxhash.h not found! Run './setup.sh'."; exit 1; }
	@echo "All dependencies are installed."

# Clean build files
clean:
	rm -f $(OUTDIR)/* deps/xxhash/xxhash.o

distclean: clean
	rm -f results/virtual/corpus/* datasets/real/* datasets/virtual/* logs/*

ultraclean: distclean
	rm -rf results datasets logs
	rm -rf deps $(OUTDIR)

.PHONY: all clean distclean ultraclean check-dependencies
