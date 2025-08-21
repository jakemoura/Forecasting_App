#!/bin/bash
# ========================================
#  Run Quarterly Outlook Forecaster (macOS)
# ========================================

echo "Launching Quarterly Outlook Forecaster..."
cd "Quarter Outlook App" || { echo "Could not find Quarter Outlook App directory."; exit 1; }
python3 outlook_forecaster.py
