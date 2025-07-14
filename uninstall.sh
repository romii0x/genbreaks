#!/bin/bash

# genbreaks uninstall script

echo "++ genbreaks uninstall ++\n"
echo ""

# get the current directory
CURRENT_DIR=$(pwd)

# check if we're in the genbreaks directory
if [[ ! -f "install.sh" ]] || [[ ! -f "demo.py" ]]; then
    echo "error: you are not in the genbreaks directory."
    echo "please run this script from the genbreaks project root."
    exit 1
fi

# confirm deletion
echo "WARNING: this will completely remove the entire genbreaks project."
echo "   this action cannot be undone."
echo ""
echo "project location: $CURRENT_DIR"
echo ""

read -p "are you sure you want to delete everything? (type 'y' to confirm): " confirmation

if [[ "$confirmation" != "y" ]]; then
    echo "uninstall cancelled."
    exit 0
fi

# remove virtual environment
if [ -d ".venv" ]; then
    echo "removing virtual environment..."
    rm -rf .venv
    echo "virtual environment removed"
fi

# go up one directory
cd ..

# remove the genbreaks directory
echo "removing entire genbreaks project..."
rm -rf "$CURRENT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "genbreaks completely removed!"
    echo "project directory deleted: $CURRENT_DIR"
else
    echo ""
    echo "error: failed to uninstall genbreaks"
    echo "you may need to manually delete: $CURRENT_DIR"
    exit 1
fi 