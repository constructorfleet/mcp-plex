import sys
from pathlib import Path

# Ensure package root is importable when tests are executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
