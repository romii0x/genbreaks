#!/bin/bash

# genbreaks runner script
# activates virtual environment and runs commands

# check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "virtual environment not found. please run ./install.sh first."
    exit 1
fi

# activate virtual environment
source .venv/bin/activate

# check if any arguments were provided
if [ $# -eq 0 ]; then
    echo "genbreaks virtual environment activated!"
    echo "run 'deactivate' to exit."
    exec bash
else
    # run the provided command
    exec "$@"
fi 