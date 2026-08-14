"""Put the repository root on ``sys.path``.

Lets the scripts run straight from a clone (``python scripts/train.py``)
without requiring an editable install first.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
