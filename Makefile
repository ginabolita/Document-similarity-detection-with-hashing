# Detect a modern compiler (use g++-13 if available, otherwise fallback to g++)
CXX := $(shell command -v g++-13 >/dev/null 2>&1 && echo g++-13 || echo g++)
FLAGS = -O3 -Wall -std=c++17
INCLUDE = -Inlohmann -Ixxhash
LIBS = deps/xxhash/libxxhash.a

EXECUTABLE_DIR = executables

# Automatically find all .cpp files and generate targets
SOURCES = $(wildcard *.cpp)
EXECUTABLES = $(patsubst %.cpp,$(EXECUTABLE_DIR)/%, $(SOURCES))

# Default target: check dependencies, build xxhash, and compile everything
all: check-dependencies deps/xxhash/libxxhash.a $(EXECUTABLES)

# Generic rule to compile any .cpp file into an executable
$(EXECUTABLE_DIR)/%: %.cpp
	@mkdir -p $(EXECUTABLE_DIR) 
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
	rm -f $(EXECUTABLES) deps/xxhash/xxhash.o

distclean: clean
	rm -f -r  ${EXECUTABLE_DIR} *.txt exp1_directory exp2_directory deps

.PHONY: all clean distclean check-dependencies
