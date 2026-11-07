#!/bin/sh
# Start GoodTerms on http://localhost:8765
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8765}
