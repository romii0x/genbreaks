"""
file utilities
"""

import os
from typing import List


def find_breakbeat_files(directory: str) -> List[str]:
    """find all wav files in directory"""
    breakbeat_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                breakbeat_files.append(file_path)
    
    return breakbeat_files 