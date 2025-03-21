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


if [ "$OS" = "Linux" ]; then
    echo "Installing Python libraries (pandas, numpy, matplotlib)..."
    $INSTALL_CMD python3-pandas python3-numpy python3-matplotlib
elif [ "$OS" = "Darwin" ]; then
    echo "Installing Python libraries (pandas, numpy, matplotlib)..."
    pip3 install pandas numpy matplotlib
else
    echo "Unsupported OS for Python library installation."
    exit 1
fi

if [! -d "deps/"]; then
    mkdir -p deps
fi

# Install nlohmann/json (header-only, no package manager needed)
if [ ! -f "deps/nlohmann/json.hpp" ]; then
    echo "Downloading nlohmann/json.hpp..."
    mkdir -p deps/nlohmann
    wget -O deps/nlohmann/json.hpp https://github.com/nlohmann/json/releases/latest/download/json.hpp
fi

# Install xxhash
if [ ! -d "xxhash" ]; then
    echo "Cloning xxhash repository..."
    git clone https://github.com/Cyan4973/xxHash.git deps/xxhash
fi

echo "All dependencies installed. You can now run 'make'."