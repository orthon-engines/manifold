"""ORTHON Dashboard Pages."""
import sys
from pathlib import Path

# Ensure streamlit directory is in path for imports
STREAMLIT_DIR = Path(__file__).parent.parent.resolve()
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))
