#!/bin/bash
# Mac / Linux launcher:  bash run_app.sh
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
