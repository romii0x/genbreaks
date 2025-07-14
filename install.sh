#!/bin/bash

# genbreaks installation script

echo "++ genbreaks installation ++\n"
echo ""

# check if python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# check python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Python version $python_version is too old. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python $python_version detected"

# check if venv module is available
if ! python3 -c "import venv" &> /dev/null; then
    echo "Python venv module not available. please install Python3-venv."
    exit 1
fi

echo "Python venv module available"

# create virtual environment
echo ""
echo "creating virtual environment..."
if [ -d ".venv" ]; then
    echo "virtual environment already exists. removing old one..."
    rm -rf .venv
fi

python3 -m venv .venv

if [ $? -eq 0 ]; then
    echo "virtual environment created"
else
    echo "failed to create virtual environment"
    exit 1
fi

# activate virtual environment
echo ""
echo "activating virtual environment..."
source .venv/bin/activate

if [ $? -eq 0 ]; then
    echo "virtual environment activated"
else
    echo "failed to activate virtual environment"
    exit 1
fi

# upgrade pip
echo ""
echo "upgrading pip..."
python -m pip install --upgrade pip

# install dependencies
echo ""
echo "installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "dependencies installed successfully"
else
    echo "failed to install dependencies"
    exit 1
fi

echo ""
echo "++ 🥁 genbreaks installed successfully! ++\n"
echo ""
echo "quick start:"
echo "1. activate the virtual environment: source .venv/bin/activate"
echo "2. run the demo: python demo.py"
echo "3. or use ./run.sh for convenience"
echo ""
echo "to deactivate the virtual environment: deactivate"
echo ""
echo "for more information, see README.md" 