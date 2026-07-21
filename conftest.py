"""
Root conftest.py — adds the project root to sys.path so that
`backend.*` imports resolve correctly when running pytest from the
project root directory.
"""

import sys
import os

# Insert the project root so `import backend.xxx` works without installing
# the package in editable mode
sys.path.insert(0, os.path.dirname(__file__))
