"""
Shared test setup. dhan_rvol.py reads DHAN_ACCESS_TOKEN/DHAN_CLIENT_ID at
IMPORT time (os.environ[...], not os.environ.get(...)) -- tests never make
real network calls to Dhan (everything is mocked), but the module still
needs these present in the environment before it can even be imported.
Set dummy values here, before any test module imports dhan_rvol or
confirm_picks (which imports dhan_rvol itself).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DHAN_ACCESS_TOKEN", "test-token")
os.environ.setdefault("DHAN_CLIENT_ID", "test-client-id")
os.environ.setdefault("GMAIL_ADDRESS", "test@example.com")
os.environ.setdefault("GMAIL_APP_PASSWORD", "test-password")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
