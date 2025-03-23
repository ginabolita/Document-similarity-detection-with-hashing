#!/bin/bash
set -e  # Stop on error

echo "Setting up project dependencies..."

# Detect OS
OS="$(uname)"
INSTALL_CMD=""

if [ "$OS" = "Linux" ]; then
    # Check if running on Debian/Ubuntu
    if command -v apt >/dev/null 2>&1; then
        INSTALL_CMD="sudo apt install -y"
    # Check if running on Fedora/Red Hat
    elif command -v yum >/dev/null 2>&1; then
        INSTALL_CMD="sudo yum install -y"
    else
        echo "Unsupported Linux distribution. Install dependencies manually."
        exit 1
    fi
elif [ "$OS" = "Darwin" ]; then
    # macOS (use Homebrew)
    if command -v brew >/dev/null 2>&1; then
        INSTALL_CMD="brew install"
    else
        echo "Homebrew is required on macOS. Install it from https://brew.sh/"
        exit 1
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Install Python and venv if not already installed
if [ "$OS" = "Linux" ]; then
    echo "Installing Python and venv..."
    $INSTALL_CMD python3 python3-venv
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created in the 'venv' directory."
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify the virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is active."
else
    echo "Failed to activate virtual environment. Please check the 'venv' directory."
    exit 1
fi

# Install Python libraries
echo "Installing Python libraries (pandas, numpy, matplotlib, seaborn)..."
pip install pandas numpy matplotlib seaborn

# Verify the libraries are installed
echo "Installed Python libraries:"
pip list | grep -E "pandas|numpy|matplotlib|seaborn"

# Create deps directory if it doesn't exist
if [ ! -d "deps/" ]; then
    echo "Creating 'deps' directory..."
    mkdir -p deps
fi

# Install nlohmann/json (header-only, no package manager needed)
if [ ! -f "deps/nlohmann/json.hpp" ]; then
    echo "Downloading nlohmann/json.hpp..."
    mkdir -p deps/nlohmann
    wget -O deps/nlohmann/json.hpp https://github.com/nlohmann/json/releases/latest/download/json.hpp
fi

# Install xxhash
if [ ! -d "deps/xxhash" ]; then
    echo "Cloning xxhash repository..."
    git clone https://github.com/Cyan4973/xxHash.git deps/xxhash
fi

echo "All dependencies installed"
echo "Running make."
make

echo "All dependencies installed"