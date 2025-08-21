#!/bin/bash
# ========================================
#  Run Main Forecasting App (macOS)
# ========================================

echo "Launching Main Forecasting App..."
cd "Forecaster App" || { echo "Could not find Forecaster App directory."; exit 1; }
python3 forecaster_app.py
