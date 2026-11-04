"""
Local Server Runner for Vercel FastAPI Application
Run: python run_local.py
Open: http://localhost:8000
"""

import sys
import os

# Set UTF-8 encoding for standard streams
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import uvicorn
from api.index import app

if __name__ == "__main__":
    print("=" * 60)
    print("AgriPrice AI Server Starting...")
    print("Local URL: http://localhost:8000")
    print("Network URL: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
