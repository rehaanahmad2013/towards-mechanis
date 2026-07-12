#!/usr/bin/env bash
set -euo pipefail
python -m pip install --quiet -r requirements.txt
python run_claim.py
